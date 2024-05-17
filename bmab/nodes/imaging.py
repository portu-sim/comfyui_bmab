import torch
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from bmab import utils
from bmab.utils import yolo
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

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/imaging'

	def process(self, image):
		image = utils.tensor2pil(image)

		net = BriaRMBG()
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

		pil_im.save('test.png')

		del net
		del img
		utils.torch_gc()

		blank = Image.new('RGBA', image.size, 0)
		blank.paste(image.convert('RGBA'), (0, 0), mask=pil_im)

		pixels = utils.pil2tensor(blank)
		return (pixels, )


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
