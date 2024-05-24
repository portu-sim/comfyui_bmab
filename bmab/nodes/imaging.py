import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

import nodes
import folder_paths

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
		results = []
		for image, processed in zip(utils.get_pils_from_pixels(target), utils.get_pils_from_pixels(source)):
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
			results.append(image)
		pixels = utils.pil2tensor(results)
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

		net = BriaRMBG()
		device = utils.get_device()
		net = BriaRMBG.from_pretrained('briaai/RMBG-1.4')
		net.to(device)
		net.eval()

		results = []
		masks = []
		for image in utils.get_pils_from_pixels(image):
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

			results.append(blank)
			masks.append(result_image)

		pixels = utils.get_pixels_from_pils(results)
		mask = utils.get_pixels_from_pils(masks)
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
		results = []
		for image1, image2 in zip(utils.get_pils_from_pixels(image1), utils.get_pils_from_pixels(image2)):
			results.append(Image.alpha_composite(image1.convert('RGBA'), image2.convert('RGBA')).convert('RGB'))
		pixels = utils.get_pixels_from_pils(results)
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
		results = []
		for image1, image2 in zip(utils.get_pils_from_pixels(image1), utils.get_pils_from_pixels(image2)):
			try:
				results.append(Image.blend(image1.convert('RGBA'), image2.convert('RGBA'), alpha=alpha).convert('RGB'))
			except ValueError:
				results.append(Image.blend(image1.convert('RGBA'), image2.resize(image1.size, Image.Resampling.LANCZOS).convert('RGBA'), alpha=alpha).convert('RGB'))
		pixels = utils.get_pixels_from_pils(results)
		return (pixels,)


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
		results = []
		for pil_img in utils.get_pils_from_pixels(image):
			masks = []
			boxes, confs = yolo.predict(pil_img, model, 0.35)
			for box in boxes:
				mask = Image.new('L', pil_img.size, 0)
				dr = ImageDraw.Draw(mask, 'L')
				x1, y1, x2, y2 = tuple(int(x) for x in box)
				x1, y1, x2, y2 = x1 - dilation, y1 - dilation, x2 + dilation, y2 + dilation
				dr.rectangle((x1, y1, x2, y2), fill=255)
				masks.append(mask)
			results.append(utils.get_pixels_from_pils(masks))
		return (results, )


class BMABLamaInpaint:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'masks': ('MASK',),
				'device': (('gpu', 'cpu'),),
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image, masks, device):
		results = []
		for image, _masks in zip(utils.get_pils_from_pixels(image), masks):
			mask = Image.new('L', image.size, 0)
			for m in utils.get_pils_from_pixels(_masks):
				mask.paste(m, (0, 0), mask=m)
			results.append(lama_inpainting(image, mask, device).convert('RGB'))
		pixels = utils.get_pixels_from_pils(results)
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
		results = []
		for pil_img in utils.get_pils_from_pixels(image):
			masks = []
			boxes, confs = yolo.predict(pil_img, model, 0.35)
			for box in boxes:
				mask = Image.new('L', pil_img.size, 0)
				dr = ImageDraw.Draw(mask, 'L')
				x1, y1, x2, y2 = tuple(int(x) for x in box)
				dr.rectangle((x1, y1, x2, y2), fill=255)
				masks.append(mask)
			results.append(utils.get_pixels_from_pils(masks))
		return (results, )


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
		results = []
		for pil_img, _masks in zip(utils.get_pils_from_pixels(image), masks):
			mask_results = []
			pil_masks = utils.get_pils_from_pixels(_masks)
			for pil_mask in pil_masks:
				box = pil_mask.getbbox()
				mask = sam.sam_predict_box(pil_img, box, model=model)
				mask_results.append(mask.convert('L'))
			results.append(utils.get_pixels_from_pils(mask_results))
		return (results, )


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
		pils = []
		for _masks in masks:
			w, h = _masks.shape[-1], _masks.shape[-2]
			mask = Image.new('L', (w, h), 0)
			for pil_mask in utils.get_pils_from_pixels(_masks):
				mask.paste(pil_mask, (0, 0), mask=pil_mask)
			pils.append(mask)
		masks = utils.get_pixels_from_pils(pils)
		for mask in masks:
			mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
		return (masks, )


class BMABLoadImage(nodes.LoadImage):

	@classmethod
	def INPUT_TYPES(s):
		input_dir = folder_paths.get_input_directory()
		files = utils.get_file_list(input_dir, input_dir)
		return {
			'required': {
				'image': (sorted(files), {'image_upload': True})
			},
		}

	CATEGORY = 'BMAB/imaging'

	RETURN_TYPES = ('IMAGE', 'MASK')
