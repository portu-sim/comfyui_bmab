import cv2
import os
import numpy as np

from PIL import Image
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from bmab import utils


bmab_model_path = os.path.join(os.path.dirname(__file__), '../../models')

sam_model = None


def sam_init(model):
	model_type = 'vit_b'
	for m in ('vit_b', 'vit_l', 'vit_h'):
		if model.find(m) >= 0:
			model_type = m
			break

	global sam_model
	if not sam_model:
		utils.lazy_loader(model)
		sam_model = sam_model_registry[model_type](checkpoint=f'%s/{model}' % bmab_model_path)
		sam_model.to(device=utils.get_device())
		sam_model.eval()
	return sam_model


def sam_predict(pilimg, boxes, model='sam_vit_b_01ec64.pth'):
	sam = sam_init(model)

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


def sam_predict_box(pilimg, box, model='sam_vit_b_01ec64.pth'):
	sam = sam_init(model)

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

	return Image.fromarray(masks[0])


def get_array_predict_box(pilimg, box, model='sam_vit_b_01ec64.pth'):
	sam = sam_init(model)
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


def release():
	global sam_model
	if sam_model is not None:
		sam_model.to(device='cpu')
	sam_model = None
	utils.torch_gc()
