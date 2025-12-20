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


class UpscaleModelWrapper:
	"""업스케일 모델 래퍼 - 자동 언로드 기능 포함"""

	def __init__(self):
		self.upscale_model = None
		self.current_model_name = None
		self.is_currently_used = False
		self.device = None

	def load_model(self, model_name):
		"""업스케일 모델 로드"""
		# 이미 같은 모델이 로드되어 있으면 재사용
		if self.upscale_model is not None and self.current_model_name == model_name:
			return self.upscale_model

		# 다른 모델이면 기존 모델 언로드
		if self.upscale_model is not None and self.current_model_name != model_name:
			print(f"Switching upscale model from {self.current_model_name} to {model_name}")
			self.unload_model()

		print(f"Loading upscale model: {model_name}")
		model_path = folder_paths.get_full_path("upscale_models", model_name)
		sd = comfy.utils.load_torch_file(model_path, safe_load=True)
		if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
			sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
		out = model_loading.load_state_dict(sd).eval()

		self.upscale_model = out
		self.current_model_name = model_name
		self.device = model_management.get_torch_device()
		print(f"Upscale model {model_name} loaded successfully")

		return self.upscale_model

	def prepare_for_inference(self, pixels):
		"""추론을 위한 메모리 확보 및 모델 GPU 로드"""
		if self.upscale_model is None:
			raise RuntimeError("Model not loaded")

		device = self.device

		# 필요한 메모리 계산
		memory_required = model_management.module_size(self.upscale_model.model)
		memory_required += (512 * 512 * 3) * pixels.element_size() * max(self.upscale_model.scale, 1.0) * 384.0
		memory_required += pixels.nelement() * pixels.element_size()

		# 메모리 확보
		model_management.free_memory(memory_required, device)

		# 모델을 GPU로 로드
		self.upscale_model.to(device)

		return device

	def unload_model(self):
		"""업스케일 모델 언로드"""
		if self.upscale_model is None:
			return

		# 사용 중이면 언로드하지 않음
		if self.is_currently_used:
			print("Upscale model is in use, skipping unload")
			return

		print(f"Unloading upscale model: {self.current_model_name}")
		self.upscale_model = None
		self.current_model_name = None

		model_management.soft_empty_cache()
		utils.torch_gc()
		print("Upscale model unloaded")

	def is_loaded(self):
		"""모델이 로드되어 있는지"""
		return self.upscale_model is not None


# 전역 업스케일 모델 관리자
class UpscaleModelManager:
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
		self._wrapper = UpscaleModelWrapper()
		self._install_hook()

	def _install_hook(self):
		"""ComfyUI의 load_models_gpu 함수에 후킹"""
		if UpscaleModelManager._hook_installed:
			return

		# 원본 함수 저장
		if not hasattr(model_management, '_original_load_models_gpu_upscale'):
			model_management._original_load_models_gpu_upscale = model_management.load_models_gpu

		# 래퍼 함수 정의
		def hooked_load_models_gpu(*args, **kwargs):
			# 업스케일 모델이 로드되어 있고 사용 중이 아니면 언로드
			if upscale_manager._wrapper.is_loaded() and not upscale_manager._wrapper.is_currently_used:
				print("ComfyUI loading other models, unloading upscale model...")
				upscale_manager._wrapper.unload_model()

			# 원본 함수 호출
			return model_management._original_load_models_gpu_upscale(*args, **kwargs)

		# 함수 교체
		model_management.load_models_gpu = hooked_load_models_gpu
		UpscaleModelManager._hook_installed = True
		print("Upscale model auto-unload hook installed")

	def get_model(self, model_name):
		"""업스케일 모델 가져오기"""
		return self._wrapper.load_model(model_name)

	def prepare_for_inference(self, pixels):
		"""추론을 위한 메모리 확보 및 GPU 로드"""
		return self._wrapper.prepare_for_inference(pixels)

	def mark_in_use(self):
		"""모델 사용 시작"""
		self._wrapper.is_currently_used = True

	def mark_not_in_use(self):
		"""모델 사용 종료"""
		self._wrapper.is_currently_used = False

	def is_loaded(self):
		"""모델이 로드되어 있는지"""
		return self._wrapper.is_loaded()

	def cleanup(self):
		"""수동 정리"""
		self._wrapper.unload_model()


# 전역 관리자 인스턴스
upscale_manager = UpscaleModelManager.get_instance()


class BMABUpscale:
	upscale_methods = ['LANCZOS', 'NEAREST', 'BILINEAR', 'BICUBIC']

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'upscale_method': (BMABUpscale.upscale_methods,),
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
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = 'upscale'

	CATEGORY = 'BMAB/upscale'

	def upscale(self, upscale_method, scale, width, height, bind: BMABBind = None, image=None):
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
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = "upscale"

	CATEGORY = "BMAB/upscale"

	def upscale_with_model(self, model_name, pixels, progress=True):
		upscale_manager.mark_in_use()
		try:
			upscale_model = upscale_manager.get_model(model_name)

			# 매니저가 메모리 관리 및 GPU 로드 처리
			device = upscale_manager.prepare_for_inference(pixels)

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

			s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
			return (s,)
		finally:
			upscale_manager.mark_not_in_use()

	def upscale(self, model_name, scale, width, height, bind: BMABBind = None, image=None):
		pixels = bind.pixels if image is None else image
		if scale != 0:
			_, h, w, c = pixels.shape
			width, height = int(w * scale), int(h * scale)

		s = self.upscale_with_model(model_name, pixels)
		pil_images = utils.get_pils_from_pixels(s)
		results = [img.resize((width, height), Image.Resampling.LANCZOS) for img in pil_images]
		pixels = utils.get_pixels_from_pils(results)

		return BMABBind.result(bind, pixels, )