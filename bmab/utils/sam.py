import cv2
import os
import numpy as np

from PIL import Image
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from bmab import utils
import comfy.model_management as mm

bmab_model_path = os.path.join(os.path.dirname(__file__), '../../models')


class SamModelWrapper:
	"""SAM 모델 래퍼 - 자동 언로드 기능 포함"""

	def __init__(self):
		self.sam_model = None
		self.is_currently_used = False

	def load_model(self, model):
		"""SAM 모델 로드"""
		if self.sam_model is not None:
			return self.sam_model

		model_type = 'vit_b'
		for m in ('vit_b', 'vit_l', 'vit_h'):
			if model.find(m) >= 0:
				model_type = m
				break

		print(f"Loading SAM model: {model}")
		utils.lazy_loader(model)
		self.sam_model = sam_model_registry[model_type](checkpoint=f'{bmab_model_path}/{model}')
		self.sam_model.to(device=mm.get_torch_device())
		self.sam_model.eval()
		print("SAM model loaded successfully")

		return self.sam_model

	def unload_model(self):
		"""SAM 모델 언로드"""
		if self.sam_model is None:
			return

		# 사용 중이면 언로드하지 않음
		if self.is_currently_used:
			print("SAM model is in use, skipping unload")
			return

		print("Unloading SAM model...")
		self.sam_model = None

		mm.soft_empty_cache()
		utils.torch_gc()
		print("SAM model unloaded")

	def is_loaded(self):
		"""모델이 로드되어 있는지"""
		return self.sam_model is not None


# 전역 SAM 모델 관리자
class SamModelManager:
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
		self._wrapper = SamModelWrapper()
		self._install_hook()

	def _install_hook(self):
		"""ComfyUI의 load_models_gpu 함수에 후킹"""
		if SamModelManager._hook_installed:
			return

		# 원본 함수 저장
		if not hasattr(mm, '_original_load_models_gpu_sam'):
			mm._original_load_models_gpu_sam = mm.load_models_gpu

		# 래퍼 함수 정의
		def hooked_load_models_gpu(*args, **kwargs):
			# SAM 모델이 로드되어 있고 사용 중이 아니면 언로드
			if sam_manager._wrapper.is_loaded() and not sam_manager._wrapper.is_currently_used:
				print("ComfyUI loading other models, unloading SAM model...")
				sam_manager._wrapper.unload_model()

			# 원본 함수 호출
			return mm._original_load_models_gpu_sam(*args, **kwargs)

		# 함수 교체
		mm.load_models_gpu = hooked_load_models_gpu
		SamModelManager._hook_installed = True
		print("SAM model auto-unload hook installed")

	def get_model(self, model='sam_vit_b_01ec64.pth'):
		"""SAM 모델 가져오기"""
		return self._wrapper.load_model(model)

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
sam_manager = SamModelManager.get_instance()


# 하위 호환성을 위한 레거시 함수들
def sam_init(model):
	"""레거시 sam_init 함수 (하위 호환성)"""
	return sam_manager.get_model(model)


def sam_predict(pilimg, boxes, model='sam_vit_b_01ec64.pth'):
	sam_manager.mark_in_use()
	try:
		sam = sam_manager.get_model(model)
		mask_predictor = SamPredictor(sam)

		numpy_image = np.array(pilimg)
		opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
		mask_predictor.set_image(opencv_image)

		result = Image.new('L', pilimg.size, 0)
		for box in boxes:
			x1, y1, x2, y2 = box

			box = np.array([int(x1), int(y1), int(x2), int(y2)])
			masks, scores, logits = mask_predictor.predict(
				box=box,
				multimask_output=False
			)

			mask = Image.fromarray(masks[0])
			result.paste(mask, mask=mask)

		return result
	finally:
		sam_manager.mark_not_in_use()


def sam_predict_box(pilimg, box, labels=None, coordinates=None, model='sam_vit_b_01ec64.pth'):
	sam_manager.mark_in_use()
	try:
		sam = sam_manager.get_model(model)
		mask_predictor = SamPredictor(sam)

		numpy_image = np.array(pilimg)
		opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
		mask_predictor.set_image(opencv_image)

		x1, y1, x2, y2 = box
		box = np.array([int(x1), int(y1), int(x2), int(y2)])

		if labels is None:
			masks, scores, logits = mask_predictor.predict(
				box=box,
				multimask_output=False
			)
		else:
			labs = np.array(labels)
			points = np.array(coordinates)
			masks, scores, logits = mask_predictor.predict(
				box=box,
				point_labels=labs,
				point_coords=points,
				multimask_output=False
			)

		return Image.fromarray(masks[0])
	finally:
		sam_manager.mark_not_in_use()


def get_array_predict_box(pilimg, box, model='sam_vit_b_01ec64.pth'):
	sam_manager.mark_in_use()
	try:
		sam = sam_manager.get_model(model)
		mask_predictor = SamPredictor(sam)

		numpy_image = np.array(pilimg)
		opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
		mask_predictor.set_image(opencv_image)

		x1, y1, x2, y2 = box
		box = np.array([int(x1), int(y1), int(x2), int(y2)])
		masks, scores, logits = mask_predictor.predict(
			box=box,
			multimask_output=False
		)

		return masks[0]
	finally:
		sam_manager.mark_not_in_use()


def release():
	"""레거시 release 함수 (하위 호환성)"""
	sam_manager.cleanup()