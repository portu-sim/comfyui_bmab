from PIL import Image
from PIL import ImageDraw

from comfy_extras.chainner_models import model_loading
from comfy import model_management
import torch
import comfy.utils
import folder_paths

import nodes
from bmab import utils
from bmab.nodes.binder import BMABBind


class BMABUpscale:
	upscale_methods = ['LANCZOS', 'NEAREST', 'BILINEAR', 'BICUBIC']

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'width': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE',)
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'upscale'

	CATEGORY = 'BMAB/upscale'

	def upscale(self, upscale_method, width, height, bind: BMABBind=None, image=None):
		pixels = bind.pixels if image is None else image
		pil_upscale_methods = {
			'LANCZOS': Image.Resampling.LANCZOS,
			'BILINEAR': Image.Resampling.BILINEAR,
			'BICUBIC': Image.Resampling.BICUBIC,
			'NEAREST': Image.Resampling.NEAREST,
		}
		bgimg = utils.tensor2pil(pixels)
		method = pil_upscale_methods.get(upscale_method)
		bgimg = bgimg.resize((width, height), method)
		pixels = utils.pil2tensor(bgimg.convert('RGB'))
		return BMABBind.result(bind, pixels, )


class BMABResizeAndFill:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'width': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
			},
			'optional': {
				'image': ('IMAGE',),
			},
		}

	RETURN_TYPES = ('IMAGE', 'MASK', )
	RETURN_NAMES = ('image', 'mask', )
	FUNCTION = 'upscale'

	CATEGORY = 'BMAB/upscale'

	def upscale(self, image, width, height):
		bgimg = utils.tensor2pil(image)

		resized = Image.new('RGB', (width, height), 0)

		mask = Image.new('L', (width, height), 0)
		dr = ImageDraw.Draw(mask, 'L')

		iratio = width / height
		cratio = bgimg.width / bgimg.height
		if iratio < cratio:
			ratio = width / bgimg.width
			w, h = int(bgimg.width * ratio), int(bgimg.height * ratio)
			y0 = (height - h) // 2
			dr.rectangle((0, y0, w, y0 + h), fill=255)
			resized.paste(bgimg.resize((w, h), Image.Resampling.LANCZOS), (0, y0))
		else:
			ratio = height / bgimg.height
			w, h = int(bgimg.width * ratio), int(bgimg.height * ratio)
			x0 = (width - w) // 2
			dr.rectangle((x0, 0, x0 + w, h), fill=255)
			resized.paste(bgimg.resize((w, h), Image.Resampling.LANCZOS), (x0, 0))

		pixels = utils.pil2tensor(resized.convert('RGB'))
		mask_pixels = utils.pil2tensor_mask(mask).unsqueeze(0)
		return (pixels, mask_pixels, )


class BMABUpscaleWithModel:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model_name": (folder_paths.get_filename_list("upscale_models"),),
				'width': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
			},
			'optional': {
				'bind': ('BMAB bind',),
				'image': ('IMAGE',),
			},
		}

	RETURN_TYPES = ('BMAB bind', "IMAGE",)
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = "upscale"

	CATEGORY = "BMAB/upscale"

	def load_model(self, model_name):
		model_path = folder_paths.get_full_path("upscale_models", model_name)
		sd = comfy.utils.load_torch_file(model_path, safe_load=True)
		if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
			sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
		out = model_loading.load_state_dict(sd).eval()
		return out

	def upscale(self, model_name, width, height, bind: BMABBind=None, image=None):
		pixels = bind.pixels if image is None else image

		device = model_management.get_torch_device()

		upscale_model = self.load_model(model_name)
		memory_required = model_management.module_size(upscale_model)
		memory_required += (512 * 512 * 3) * pixels.element_size() * max(upscale_model.scale, 1.0) * 384.0  # The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
		memory_required += pixels.nelement() * pixels.element_size()
		model_management.free_memory(memory_required, device)

		upscale_model.to(device)
		in_img = pixels.movedim(-1, -3).to(device)

		tile = 512
		overlap = 32

		oom = True
		while oom:
			try:
				steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
				pbar = comfy.utils.ProgressBar(steps)
				s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
				oom = False
			except model_management.OOM_EXCEPTION as e:
				tile //= 2
				if tile < 128:
					raise e

		upscale_model.cpu()
		s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

		bgimg = utils.tensor2pil(s)
		bgimg = bgimg.resize((width, height), Image.Resampling.LANCZOS)
		pixels = utils.pil2tensor(bgimg.convert('RGB'))

		return BMABBind.result(bind, pixels,)
