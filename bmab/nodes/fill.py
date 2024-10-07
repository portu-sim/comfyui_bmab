import math
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download

from bmab.external.fill.controlnet_union import ControlNetModel_Union
from bmab.external.fill.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from PIL import Image, ImageDraw

from bmab import utils


pipe = None


def load():
	global pipe

	config_file = hf_hub_download(
		"xinsir/controlnet-union-sdxl-1.0",
		filename="config_promax.json",
	)

	config = ControlNetModel_Union.load_config(config_file)
	controlnet_model = ControlNetModel_Union.from_config(config)
	model_file = hf_hub_download(
		"xinsir/controlnet-union-sdxl-1.0",
		filename="diffusion_pytorch_model_promax.safetensors",
	)
	state_dict = load_state_dict(model_file)
	model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
		controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
	)
	model.to(device="cuda", dtype=torch.float16)

	vae = AutoencoderKL.from_pretrained(
		"madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
	).to("cuda")

	pipe = StableDiffusionXLFillPipeline.from_pretrained(
		"SG161222/RealVisXL_V5.0_Lightning",
		torch_dtype=torch.float16,
		vae=vae,
		controlnet=model,
		variant="fp16",
	).to("cuda")

	pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)


class BMABReframe:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'ratio': (['1:1', '4:5', '2:3', '9:16', '5:4', '3:2', '16:9'],),
				'dilation': ('INT', {'default': 32, 'min': 4, 'max': 128, 'step': 1}),
				'step': ('INT', {'default': 8, 'min': 4, 'max': 128, 'step': 1}),
				'iteration': ('INT', {'default': 4, 'min': 1, 'max': 8, 'step': 1}),
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
			}
		}

	RETURN_TYPES = ('IMAGE',)
	RETURN_NAMES = ('image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/fill'

	ratio_sel = {
		'1:1': (1024, 1024),
		'4:5': (960, 1200),
		'2:3': (896, 1344),
		'9:16': (816, 1456),
		'5:4': (1200, 960),
		'3:2': (1344, 896),
		'16:9': (1456, 816)
	}

	def infer(self, image, width, height, overlap_width, num_inference_steps, prompt_input):
		source = image
		image_ratio = source.width / source.height
		output_ratio = width / height

		if output_ratio <= image_ratio:
			ratio = width / source.width
		else:
			ratio = height / source.height

		source = source.resize((math.ceil(source.width * ratio), math.ceil(source.height * ratio)), Image.Resampling.LANCZOS)
		background = Image.new('RGB', (width, height), (255, 255, 255))
		mask = Image.new('L', (width, height), 255)
		mask_draw = ImageDraw.Draw(mask)

		if output_ratio <= image_ratio:
			margin = (height - source.height) // 2
			background.paste(source, (0, margin))
			mask_draw.rectangle((0, margin + overlap_width, source.width, margin + source.height - overlap_width), fill=0)
		else:
			margin = (width - source.width) // 2
			background.paste(source, (margin, 0))
			mask_draw.rectangle((margin + overlap_width, 0, margin + source.width - overlap_width, source.height), fill=0)

		cnet_image = background.copy()
		cnet_image.paste(0, (0, 0), mask)

		final_prompt = f"{prompt_input} , high quality, 4k"

		if pipe is None:
			load()

		(
			prompt_embeds,
			negative_prompt_embeds,
			pooled_prompt_embeds,
			negative_pooled_prompt_embeds,
		) = pipe.encode_prompt(final_prompt, "cuda", True)

		image = pipe(
				prompt_embeds=prompt_embeds,
				negative_prompt_embeds=negative_prompt_embeds,
				pooled_prompt_embeds=pooled_prompt_embeds,
				negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
				image=cnet_image,
				num_inference_steps=num_inference_steps
			)

		image = image.convert("RGBA")
		cnet_image.paste(image, (0, 0), mask)

		return cnet_image

	def process(self, image, ratio, dilation, step, iteration, prompt, **kwargs):

		r = BMABReframe.ratio_sel.get(ratio, (1024, 1024))

		results = []
		for image in utils.get_pils_from_pixels(image):
			for v in range(0, iteration):
				a = self.infer(image, r[0], r[1], dilation, step, prompt_input=prompt)
				results.append(a)
		pixels = utils.get_pixels_from_pils(results)

		return (pixels,)


class BMABOutpaintByRatio:
	resize_methods = ['stretching', 'inpaint', 'inpaint+lama']
	resize_alignment = ['bottom', 'top', 'top-right', 'right', 'bottom-right', 'bottom-left', 'left', 'top-left', 'center']

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'steps': ('INT', {'default': 8, 'min': 0, 'max': 10000}),
				'alignment': (s.resize_alignment,),
				'ratio': ('FLOAT', {'default': 0.85, 'min': 0.1, 'max': 0.95, 'step': 0.01}),
				'dilation': ('INT', {'default': 32, 'min': 4, 'max': 128, 'step': 1}),
				'iteration': ('INT', {'default': 4, 'min': 1, 'max': 8, 'step': 1}),
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
			},
			'optional': {
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/fill'

	@staticmethod
	def image_alignment(image, left, right, top, bottom, ratio):
		left = int(left)
		top = int(top)
		input_image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS)
		background = Image.new('RGB', image.size, (255, 255, 255))
		background.paste(input_image, box=(left, top))
		return background

	@staticmethod
	def mask_alignment(width, height, left, right, top, bottom, ratio, dilation):
		left = int(left)
		top = int(top)
		w, h = math.ceil(width * ratio), math.ceil(height * ratio)
		mask = Image.new('L', (width, height), 255)
		mask_draw = ImageDraw.Draw(mask)
		box = (
			0 if left == 0 else left + dilation,
			0 if top == 0 else top + dilation,
			width if (left + w) >= width else (left + w - dilation),
			height if (top + h) >= height else (top + h - dilation)
		)
		mask_draw.rectangle(box, fill=0)
		return mask

	def infer(self, image, al, ratio, dilation, num_inference_steps, prompt_input):
		if al not in utils.alignment:
			return image
		w, h = math.ceil(image.width * (1 - ratio)), math.ceil(image.height * (1 - ratio))
		background = BMABOutpaintByRatio.image_alignment(image, *utils.alignment[al](w, h), ratio)
		mask = BMABOutpaintByRatio.mask_alignment(image.width, image.height, *utils.alignment[al](w, h), ratio, dilation)

		cnet_image = background.copy()
		cnet_image.paste(0, (0, 0), mask)

		final_prompt = f"{prompt_input} , high quality, 4k"

		if pipe is None:
			load()

		(
			prompt_embeds,
			negative_prompt_embeds,
			pooled_prompt_embeds,
			negative_pooled_prompt_embeds,
		) = pipe.encode_prompt(final_prompt, "cuda", True)

		image = pipe(
			prompt_embeds=prompt_embeds,
			negative_prompt_embeds=negative_prompt_embeds,
			pooled_prompt_embeds=pooled_prompt_embeds,
			negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
			image=cnet_image,
			num_inference_steps=num_inference_steps
		)

		image = image.convert("RGBA")
		cnet_image.paste(image, (0, 0), mask)

		return cnet_image

	def process(self, image, steps, alignment, ratio, dilation, iteration, prompt):

		results = []
		for image in utils.get_pils_from_pixels(image):

			print('Process image resize', ratio)
			for r in range(0, iteration):
				a = self.infer(image, alignment, ratio, dilation, steps, prompt_input=prompt)
				results.append(a)

		pixels = utils.get_pixels_from_pils(results)
		return (pixels,)



class BMABInpaint:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'mask': ('MASK',),
				'steps': ('INT', {'default': 8, 'min': 0, 'max': 10000}),
				'iteration': ('INT', {'default': 4, 'min': 1, 'max': 8, 'step': 1}),
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
			},
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/fill'

	def infer(self, image, mask, steps, prompt_input):

		source = image
		source.paste((255, 255, 255), (0, 0), mask)

		cnet_image = source.copy()
		cnet_image.paste(0, (0, 0), mask)

		final_prompt = f"{prompt_input} , high quality, 4k"

		if pipe is None:
			load()

		(
			prompt_embeds,
			negative_prompt_embeds,
			pooled_prompt_embeds,
			negative_pooled_prompt_embeds,
		) = pipe.encode_prompt(final_prompt, "cuda", True)

		image = pipe(
			prompt_embeds=prompt_embeds,
			negative_prompt_embeds=negative_prompt_embeds,
			pooled_prompt_embeds=pooled_prompt_embeds,
			negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
			image=cnet_image,
			num_inference_steps=steps
		)

		image = image.convert("RGBA")
		if image.size != mask.size:
			image = image.resize(mask.size, Image.Resampling.LANCZOS)
		cnet_image.paste(image, (0, 0), mask)

		return cnet_image

	def mask_to_image(self, mask):
		result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
		return utils.get_pils_from_pixels(result)[0].convert('L')

	def process(self, image, mask, steps, iteration, prompt):

		results = []
		mask = self.mask_to_image(mask)
		for image in utils.get_pils_from_pixels(image):
			for r in range(0, iteration):
				a = self.infer(image, mask, steps, prompt_input=prompt)
				results.append(a)

		pixels = utils.get_pixels_from_pils(results)
		return (pixels,)

