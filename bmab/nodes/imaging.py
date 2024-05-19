import torch
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from bmab import utils
from bmab.utils import yolo, sam
from bmab.external.rmbg14.briarmbg import BriaRMBG
from bmab.external.rmbg14.utilities import preprocess_image, postprocess_image
from bmab.external.lama import lama_inpainting


class BMABDetectionCrop:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'source': ('IMAGE',),
				'target': ('IMAGE',),
				'model': (utils.list_pretraining_models(),),
				'padding': ('INT', {'default': 32, 'min': 8, 'max': 128, 'step': 8}),
				'dilation': ('INT', {'default': 4, 'min': 4, 'max': 32, 'step': 1}),
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, source, target, model, padding, dilation):
		processed = utils.tensor2pil(source)
		image = utils.tensor2pil(target)

		dup = image.copy()
		boxes, confs = yolo.predict(processed, model, 0.35)
		for box in boxes:
			x1, y1, x2, y2 = tuple(int(x) for x in box)
			x1, y1, x2, y2 = x1 - dilation, y1 - dilation, x2 + dilation, y2 + dilation

			mask = Image.new('L', processed.size, 0)
			dr = ImageDraw.Draw(mask, 'L')
			dr.rectangle((x1, y1, x2, y2), fill=255)

			padding_box = x1 - padding, y1 - padding, x2 + padding, y2 + padding
			cropped = processed.crop(padding_box)
			dup.paste(cropped, (padding_box[0], padding_box[1]))

			blur = ImageFilter.GaussianBlur(5)
			mask = mask.filter(blur)

			image.paste(dup, (0, 0), mask=mask)

		pixels = utils.pil2tensor(image.convert('RGB'))
		return (pixels, )


class BMABRemoveBackground:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('IMAGE', 'MASK', )
	RETURN_NAMES = ('image', 'MASK', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image):
		image = utils.tensor2pil(image)

		net = BriaRMBG()
		device = utils.get_device()
		net = BriaRMBG.from_pretrained('briaai/RMBG-1.4')
		net.to(device)
		net.eval()

		model_input_size = [image.width, image.height]
		orig_im = np.array(image)
		orig_im_size = orig_im.shape[0:2]
		img = preprocess_image(orig_im, model_input_size).to(device)

		# inference
		result = net(img)

		# post process
		result_image = postprocess_image(result[0][0], orig_im_size)

		# save result
		pil_im = Image.fromarray(result_image)

		del net
		del img
		utils.torch_gc()

		blank = Image.new('RGBA', image.size, 0)
		blank.paste(image.convert('RGBA'), (0, 0), mask=pil_im)

		pixels = utils.pil2tensor(blank)
		mask = utils.pil2tensor_mask(result_image)
		return (pixels, mask, )


class BMABAlphaComposit:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image1': ('IMAGE',),
				'image2': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image1, image2):
		image1 = utils.tensor2pil(image1).convert('RGBA')
		image2 = utils.tensor2pil(image2).convert('RGBA')
		image3 = Image.alpha_composite(image1, image2)
		pixels = utils.pil2tensor(image3)
		return (pixels, )


class BMABBlend:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image1': ('IMAGE',),
				'image2': ('IMAGE',),
				'alpha': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image1, image2, alpha):
		image1 = utils.tensor2pil(image1).convert('RGBA')
		image2 = utils.tensor2pil(image2).convert('RGBA')
		image3 = Image.blend(image1, image2, alpha=alpha)
		pixels = utils.pil2tensor(image3)
		return (pixels, )


class BMABDetectAndMask:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'model': (utils.list_pretraining_models(),),
				'dilation': ('INT', {'default': 4, 'min': 4, 'max': 128, 'step': 1}),
			}
		}

	RETURN_TYPES = ('MASK', )
	RETURN_NAMES = ('mask', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image, model, dilation):
		image = utils.tensor2pil(image)

		mask = Image.new('L', image.size, 0)
		dr = ImageDraw.Draw(mask, 'L')

		boxes, confs = yolo.predict(image, model, 0.35)
		for box in boxes:
			x1, y1, x2, y2 = tuple(int(x) for x in box)
			x1, y1, x2, y2 = x1 - dilation, y1 - dilation, x2 + dilation, y2 + dilation
			dr.rectangle((x1, y1, x2, y2), fill=255)
		pixels = utils.pil2tensor_mask(mask).unsqueeze(0)
		return (pixels, )


class BMABLamaInpaint:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'mask': ('MASK',),
				'device': (('gpu', 'cpu'),),
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image, mask, device):
		image = utils.tensor2pil(image)
		mask = utils.tensor2pil(mask)
		image = lama_inpainting(image, mask, device)
		pixels = utils.pil2tensor(image)
		return (pixels, )


class BMABDetector:

	@classmethod
	def INPUT_TYPES(s):
		detectors = [
			'face_yolov8n.pt', 'face_yolov8n_v2.pt', 'face_yolov8m.pt', 'face_yolov8s.pt',
			'hand_yolov8n.pt', 'hand_yolov8s.pt', 'person_yolov8m-seg.pt', 'person_yolov8n-seg.pt', 'person_yolov8s-seg.pt'
		]
		detector_models_in_path = utils.list_pretraining_models()
		detectors.extend([m for m in detector_models_in_path if m not in detectors])

		return {
			'required': {
				'image': ('IMAGE',),
				'model': (detectors, ),
			},
		}

	RETURN_TYPES = ('MASK', )
	RETURN_NAMES = ('masks', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image, model):
		pil_img = utils.tensor2pil(image)
		masks = []
		boxes, confs = yolo.predict(pil_img, model, 0.35)
		for box in boxes:
			mask = Image.new('L', pil_img.size, 0)
			dr = ImageDraw.Draw(mask, 'L')
			x1, y1, x2, y2 = tuple(int(x) for x in box)
			dr.rectangle((x1, y1, x2, y2), fill=255)
			masks.append(np.array(mask).astype(np.float32) / 255.0)
		return (torch.from_numpy(np.array(masks)), )


class BMABSegmentAnything:

	@classmethod
	def INPUT_TYPES(s):
		sam_model = ['sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth', 'sam_vit_h_4b8939.pth']

		return {
			'required': {
				'image': ('IMAGE',),
				'masks': ('MASK',),
				'model': (sam_model,),
			}
		}

	RETURN_TYPES = ('MASK',)
	RETURN_NAMES = ('masks',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image, model, masks=None):
		pil_img = utils.tensor2pil(image)

		results = []
		for (batch_number, image) in enumerate(masks):
			i = 255. * image.cpu().numpy().squeeze()
			pil_mask = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
			box = pil_mask.getbbox()
			mask = sam.get_array_predict_box(pil_img, box, model=model)
			results.append(mask)
		return (torch.from_numpy(np.array(results)), )


class BMABMasksToImages:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'masks': ('MASK',),
			}
		}

	RETURN_TYPES = ('IMAGE',)
	RETURN_NAMES = ('images',)
	FUNCTION = 'mask_to_image'

	CATEGORY = 'BMAB/imaging'


	def mask_to_image(self, masks):
		for (batch_number, mask) in enumerate(masks):
			mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
		return (masks, )


