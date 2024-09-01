import os.path

import numpy as np
import cv2
import torch
import hashlib

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageSequence

import nodes
import folder_paths
import node_helpers

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
				'channel': (['RGBA', 'RGB'],),
			}
		}

	RETURN_TYPES = ('IMAGE', 'MASK', )
	RETURN_NAMES = ('image', 'MASK', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image, channel):

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

			if channel == 'RGBA':
				blank = Image.new('RGBA', image.size, 0)
				blank.paste(image.convert('RGBA'), (0, 0), mask=pil_im)
				results.append(blank)
			elif channel == 'RGB':
				blank = Image.new('RGB', image.size, 0)
				blank.paste(image.convert('RGB'), (0, 0), mask=pil_im)
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
			},
			'optional': {
				'alpha': ('MASK',),
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image1, image2, alpha):
		results = []
		for image1, image2 in zip(utils.get_pils_from_pixels(image1), utils.get_pils_from_pixels(image2)):
			if alpha is not None:
				alpha = utils.tensor2pil(alpha)
				image2.putalpha(alpha)
			try:
				results.append(Image.alpha_composite(image1.convert('RGBA'), image2.convert('RGBA')).convert('RGB'))
			except ValueError:
				results.append(Image.alpha_composite(image1.convert('RGBA'), image2.resize(image1.size, Image.Resampling.LANCZOS).convert('RGBA')).convert('RGB'))
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


class BMABDetectAndPaste:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'source': ('IMAGE',),
				'model': (utils.list_pretraining_models(),),
				'dilation': ('INT', {'default': 4, 'min': 4, 'max': 128, 'step': 1}),
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image, source, model, dilation):
		results = []
		src = utils.get_pils_from_pixels(source)
		for pil_img in utils.get_pils_from_pixels(image):
			boxes, confs = yolo.predict(src[0], model, 0.35)
			for box, conf in zip(boxes, confs):
				x1, y1, x2, y2 = tuple(int(x) for x in box)
				pil_img.paste(src[0], (0, 0), mask=utils.get_blur_mask(pil_img.size, (x1, y1, x2, y2), dilation))
			results.append(pil_img)
		return (utils.get_pixels_from_pils(results), )


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

		if not isinstance(masks, list):
			masks.reshape((-1, 1, masks.shape[-2], masks.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
		else:
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


class BMABLoadOutputImage:

	@classmethod
	def INPUT_TYPES(s):
		output_dir = folder_paths.get_output_directory()
		files = utils.get_file_list(output_dir, output_dir)
		return {
			'required': {
				'image': (sorted(files), {'image_upload': False})
			},
		}

	CATEGORY = 'BMAB/imaging'

	RETURN_TYPES = ('IMAGE', 'MASK')
	FUNCTION = "load_image"

	def load_image(self, image):
		image_path = os.path.join(folder_paths.get_output_directory(), image)
		img = node_helpers.pillow(Image.open, image_path)

		output_images = []
		output_masks = []
		w, h = None, None

		excluded_formats = ['MPO']

		for i in ImageSequence.Iterator(img):
			i = node_helpers.pillow(ImageOps.exif_transpose, i)

			if i.mode == 'I':
				i = i.point(lambda i: i * (1 / 255))
			image = i.convert("RGB")

			if len(output_images) == 0:
				w = image.size[0]
				h = image.size[1]

			if image.size[0] != w or image.size[1] != h:
				continue

			image = np.array(image).astype(np.float32) / 255.0
			image = torch.from_numpy(image)[None,]
			if 'A' in i.getbands():
				mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
				mask = 1. - torch.from_numpy(mask)
			else:
				mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
			output_images.append(image)
			output_masks.append(mask.unsqueeze(0))

		if len(output_images) > 1 and img.format not in excluded_formats:
			output_image = torch.cat(output_images, dim=0)
			output_mask = torch.cat(output_masks, dim=0)
		else:
			output_image = output_images[0]
			output_mask = output_masks[0]

		return (output_image, output_mask)


	@classmethod
	def IS_CHANGED(s, image):
		image_path = os.path.join(folder_paths.get_output_directory(), image)
		m = hashlib.sha256()
		with open(image_path, 'rb') as f:
			m.update(f.read())
		return m.digest().hex()


	@classmethod
	def VALIDATE_INPUTS(s, image):
		image_path = os.path.join(folder_paths.get_output_directory(), image)
		if not os.path.exists(image_path):
			return "Invalid image file: {}".format(image)

		return True


class BMABEdge:
	@classmethod
	def INPUT_TYPES(s):
		return {'required': {
			'pixels': ('IMAGE',),
			'threshold1': ('FLOAT', {'default': 50.0, 'min': 1.0, 'max': 255, 'step': 1}),
			'threshold2': ('FLOAT', {'default': 200.0, 'min': 1.0, 'max': 255, 'step': 1}),
			'strength': ('FLOAT', {'default': 0.5, 'min': 0, 'max': 1.0, 'step': 0.05}),
		},
			'hidden': {'unique_id': 'UNIQUE_ID'}
		}

	RETURN_TYPES = ('IMAGE',)
	RETURN_NAMES = ('image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/basic'

	@staticmethod
	def edge_flavor(pil, canny_th1: int, canny_th2: int, strength: float):
		numpy_image = np.array(pil)
		base = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
		arcanny = cv2.Canny(base, canny_th1, canny_th2)
		canny = Image.fromarray(arcanny)
		canny = ImageOps.invert(canny)

		mdata = canny.getdata()
		ndata = pil.getdata()

		newdata = []
		for idx in range(0, len(mdata)):
			if mdata[idx] == 0:
				newdata.append((0, 0, 0))
			else:
				newdata.append(ndata[idx])

		newbase = Image.new('RGB', pil.size)
		newbase.putdata(newdata)
		return Image.blend(pil, newbase, alpha=strength).convert('RGB')

	def process(self, pixels, threshold1, threshold2, strength, unique_id):
		results = []
		for bgimg in utils.get_pils_from_pixels(pixels):
			bgimg = self.edge_flavor(bgimg, threshold1, threshold2, strength)
			results.append(bgimg)
		pixels = utils.pil2tensor(results)
		return (pixels,)


class BMABBlackAndWhite:

	@classmethod
	def INPUT_TYPES(s):

		return {
			'required': {
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('IMAGE',)
	RETURN_NAMES = ('image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image):
		results = []
		for pil_img in utils.get_pils_from_pixels(image):
			l_mode = pil_img.convert('L')
			thresh = 200
			fn = lambda x: 255 if x > thresh else 0
			pil = l_mode.point(fn, mode='1').convert('RGB')
			results.append(pil)
		pixels = utils.get_pixels_from_pils(results)
		return (pixels,)
