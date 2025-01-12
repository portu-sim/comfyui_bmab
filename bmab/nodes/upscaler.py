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
				'upscale_method': (BMABUpscale.upscale_methods, ),
				'scale': ('FLOAT', {'default': 2.0, 'min': 0, 'max': 4.0, 'step': 0.001}),
				'width': ('INT', {'default': 512, 'min': 32, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 32, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
			},
			'optional': {
				'bind': ('BMAB bind',),
				'image': ('IMAGE',),
			},
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE',)
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'upscale'

	CATEGORY = 'BMAB/upscale'

	def upscale(self, upscale_method, scale, width, height, bind: BMABBind=None, image=None):
		pixels = bind.pixels if image is None else image
		pil_upscale_methods = {
			'LANCZOS': Image.Resampling.LANCZOS,
			'BILINEAR': Image.Resampling.BILINEAR,
			'BICUBIC': Image.Resampling.BICUBIC,
			'NEAREST': Image.Resampling.NEAREST,
		}
		results = []
		for bgimg in utils.get_pils_from_pixels(pixels):
			if scale != 0:
				width, height = int(bgimg.width * scale), int(bgimg.height * scale)
			method = pil_upscale_methods.get(upscale_method)
			results.append(bgimg.resize((width, height), method))
		pixels = utils.get_pixels_from_pils(results)
		return BMABBind.result(bind, pixels, )


class BMABUpscaleWithModel:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model_name": (folder_paths.get_filename_list("upscale_models"),),
				'scale': ('FLOAT', {'default': 2.0, 'min': 0, 'max': 4.0, 'step': 0.001}),
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

	def upscale_with_model(self, model_name, pixels, progress=True):
		upscale_model = self.load_model(model_name)
		device = model_management.get_torch_device()

		memory_required = model_management.module_size(upscale_model.model)
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
				if progress:
					steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
					pbar = comfy.utils.ProgressBar(steps)
					s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
				else:
					s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale)
				oom = False
			except model_management.OOM_EXCEPTION as e:
				tile //= 2
				if tile < 128:
					raise e

		upscale_model.to("cpu")
		s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
		return (s,)

	def upscale(self, model_name, scale, width, height, bind: BMABBind=None, image=None):
		pixels = bind.pixels if image is None else image
		if scale != 0:
			_, h, w, c = pixels.shape
			width, height = int(w * scale), int(h * scale)

		s = self.upscale_with_model(model_name, pixels)
		pil_images = utils.get_pils_from_pixels(s)
		results = [img.resize((width, height), Image.Resampling.LANCZOS) for img in pil_images]
		pixels = utils.get_pixels_from_pils(results)

		return BMABBind.result(bind, pixels,)
