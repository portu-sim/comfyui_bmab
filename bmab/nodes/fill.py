import math
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download

from bmab.external.fill.controlnet_union import ControlNetModel_Union
from bmab.external.fill.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from PIL import Image, ImageDraw, ImageFilter

from bmab import utils
import comfy.model_management as mm


class FillPipelineWrapper:
	"""파이프라인 래퍼 - 자동 언로드 기능 포함"""

	def __init__(self):
		self.pipe = None
		self.device = None
		self.memory_required = 8 * 1024 * 1024 * 1024  # 8GB
		self.is_currently_used = False

	def load_pipeline(self):
		"""파이프라인 로드"""
		if self.pipe is not None:
			return self.pipe

		print("Loading fill pipeline...")

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
		try:
			model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
				controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
			)
		except:
			# diffuers >= 0.44
			model, _, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
				controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0", []
			)

		self.device = mm.get_torch_device()
		model.to(device=self.device, dtype=torch.float16)

		vae = AutoencoderKL.from_pretrained(
			"madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
		).to(self.device)

		self.pipe = StableDiffusionXLFillPipeline.from_pretrained(
			"SG161222/RealVisXL_V5.0_Lightning",
			torch_dtype=torch.float16,
			vae=vae,
			controlnet=model,
			variant="fp16",
		).to(self.device)

		self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

		print("Fill pipeline loaded successfully")
		return self.pipe

	def unload_pipeline(self):
		"""파이프라인 언로드"""
		if self.pipe is None:
			return

		# 사용 중이면 언로드하지 않음
		if self.is_currently_used:
			print("Fill pipeline is in use, skipping unload")
			return

		print("Unloading fill pipeline...")

		# 모든 서브모델을 CPU로 이동
		if hasattr(self.pipe, 'unet') and self.pipe.unet is not None:
			self.pipe.unet.to('cpu')
		if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
			self.pipe.vae.to('cpu')
		if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
			self.pipe.text_encoder.to('cpu')
		if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None:
			self.pipe.text_encoder_2.to('cpu')
		if hasattr(self.pipe, 'controlnet') and self.pipe.controlnet is not None:
			self.pipe.controlnet.to('cpu')

		self.pipe = None

		# 메모리 정리
		mm.soft_empty_cache()
		utils.torch_gc()

		print("Fill pipeline unloaded")

	def is_loaded(self):
		"""파이프라인이 로드되어 있는지"""
		return self.pipe is not None


# 전역 파이프라인 관리자
class PipelineManager:
	_instance = None
	_wrapper = None
	_original_load_models_gpu = None
	_hook_installed = False

	@classmethod
	def get_instance(cls):
		if cls._instance is None:
			cls._instance = cls()
		return cls._instance

	def __init__(self):
		self._wrapper = FillPipelineWrapper()
		self._install_hook()

	def _install_hook(self):
		"""ComfyUI의 load_models_gpu 함수에 후킹"""
		if PipelineManager._hook_installed:
			return

		# 원본 함수 저장
		PipelineManager._original_load_models_gpu = mm.load_models_gpu

		# 래퍼 함수 정의
		def hooked_load_models_gpu(*args, **kwargs):
			# Fill 파이프라인이 로드되어 있고 사용 중이 아니면 언로드
			if pipe_manager._wrapper.is_loaded() and not pipe_manager._wrapper.is_currently_used:
				print("ComfyUI loading other models, unloading fill pipeline...")
				pipe_manager._wrapper.unload_pipeline()

			# 원본 함수 호출
			return PipelineManager._original_load_models_gpu(*args, **kwargs)

		# 함수 교체
		mm.load_models_gpu = hooked_load_models_gpu
		PipelineManager._hook_installed = True
		print("Fill pipeline auto-unload hook installed")

	def get_pipe(self):
		"""파이프라인 가져오기"""
		return self._wrapper.load_pipeline()

	def mark_in_use(self):
		"""파이프라인 사용 시작"""
		self._wrapper.is_currently_used = True

	def mark_not_in_use(self):
		"""파이프라인 사용 종료"""
		self._wrapper.is_currently_used = False

	def is_loaded(self):
		"""파이프라인이 로드되어 있는지"""
		return self._wrapper.is_loaded()

	def cleanup(self):
		"""수동 정리"""
		self._wrapper.unload_pipeline()


# 하위 호환성을 위한 레거시 함수
pipe_manager = PipelineManager.get_instance()


def load():
	"""레거시 load 함수"""
	return pipe_manager.get_pipe()


def unload():
	"""레거시 unload 함수"""
	pipe_manager.cleanup()


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

		# 파이프라인 사용 시작
		pipe_manager.mark_in_use()
		try:
			pipe = pipe_manager.get_pipe()

			(
				prompt_embeds,
				negative_prompt_embeds,
				pooled_prompt_embeds,
				negative_pooled_prompt_embeds,
			) = pipe.encode_prompt(final_prompt, pipe.device, True)

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
		finally:
			# 파이프라인 사용 종료
			pipe_manager.mark_not_in_use()

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

	RETURN_TYPES = ('IMAGE',)
	RETURN_NAMES = ('image',)
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

		# 파이프라인 사용 시작
		pipe_manager.mark_in_use()
		try:
			pipe = pipe_manager.get_pipe()

			(
				prompt_embeds,
				negative_prompt_embeds,
				pooled_prompt_embeds,
				negative_pooled_prompt_embeds,
			) = pipe.encode_prompt(final_prompt, pipe.device, True)

			image = pipe(
				prompt_embeds=prompt_embeds,
				negative_prompt_embeds=negative_prompt_embeds,
				pooled_prompt_embeds=pooled_prompt_embeds,
				negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
				image=cnet_image,
				num_inference_steps=num_inference_steps
			)

			return image
		finally:
			# 파이프라인 사용 종료
			pipe_manager.mark_not_in_use()

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
			'optional': {
				'seed': ('SEED',)
			}
		}

	RETURN_TYPES = ('IMAGE',)
	RETURN_NAMES = ('image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/fill'

	def infer(self, image, mask, steps, prompt_input):

		source = image
		source.paste((255, 255, 255), (0, 0), mask)

		cnet_image = source.copy()
		cnet_image.paste(0, (0, 0), mask)

		final_prompt = f"{prompt_input} , high quality, 4k"

		# 파이프라인 사용 시작
		pipe_manager.mark_in_use()
		try:
			pipe = pipe_manager.get_pipe()

			(
				prompt_embeds,
				negative_prompt_embeds,
				pooled_prompt_embeds,
				negative_pooled_prompt_embeds,
			) = pipe.encode_prompt(final_prompt, pipe.device, True)

			image = pipe(
				prompt_embeds=prompt_embeds,
				negative_prompt_embeds=negative_prompt_embeds,
				pooled_prompt_embeds=pooled_prompt_embeds,
				negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
				image=cnet_image,
				num_inference_steps=steps
			)
			return image
		finally:
			# 파이프라인 사용 종료
			pipe_manager.mark_not_in_use()

	def mask_to_image(self, mask):
		result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
		return utils.get_pils_from_pixels(result)[0].convert('L')

	def process(self, image, mask, steps, iteration, prompt, seed=None):
		results = []
		mask = self.mask_to_image(mask)
		for image in utils.get_pils_from_pixels(image):
			for r in range(0, iteration):
				a = self.infer(image, mask, steps, prompt_input=prompt)
				results.append(a)

		pixels = utils.get_pixels_from_pils(results)
		return (pixels,)