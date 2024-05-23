import comfy
import nodes

from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFilter
from ultralytics import YOLO
from bmab import utils
from bmab.external.lama import LamaInpainting
from bmab.nodes.binder import BMABBind
from bmab import process


def predict(image: Image, model, confidence):
	yolo = utils.lazy_loader(model)
	boxes = []
	confs = []
	try:
		model = YOLO(yolo)
		pred = model(image, conf=confidence, device='')
		boxes = pred[0].boxes.xyxy.cpu().numpy()
		boxes = boxes.tolist()
		confs = pred[0].boxes.conf.tolist()
	except:
		pass
	utils.torch_gc()
	return boxes, confs


class BMABResizeByPerson:
	resize_methods = ['stretching', 'inpaint', 'inpaint+lama']
	resize_alignment = ['bottom', 'top', 'top-right', 'right', 'bottom-right', 'bottom-left', 'left', 'top-left', 'center']

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'method': (s.resize_methods,),
				'alignment': (s.resize_alignment,),
				'ratio': ('FLOAT', {'default': 0.85, 'min': 0.6, 'max': 0.95, 'step': 0.01}),
				'dilation': ('INT', {'default': 30, 'min': 4, 'max': 128, 'step': 1}),
			},
			'optional': {
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE', )
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/resize'

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, method, alignment, ratio, dilation, image=None):
		pixels = bind.pixels if image is None else image

		results = []
		for image in utils.get_pils_from_pixels(pixels):

			if bind.context is not None:
				steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

			boxes, confs = predict(image, 'person_yolov8m-seg.pt', 0.35)
			if len(boxes) == 0:
				results.append(image.convert('RGB'))
				continue

			largest = []
			for idx, box in enumerate(boxes):
				x1, y1, x2, y2 = box
				largest.append(((y2 - y1), box))
			largest = sorted(largest, key=lambda c: c[0], reverse=True)

			x1, y1, x2, y2 = largest[0][1]
			pratio = (y2 - y1) / image.height
			if pratio > ratio:
				image_ratio = pratio / ratio
				if image_ratio < 1.0:
					results.append(image.convert('RGB'))
					continue
			else:
				results.append(image.convert('RGB'))
				continue

			stretching_image = utils.resize_image_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio))
			if method == 'stretching':
				results.append(stretching_image.convert('RGB'))
			elif method == 'inpaint':
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio), dilation)
				blur = ImageFilter.GaussianBlur(10)
				blur_mask = mask.filter(blur)
				blur_mask = ImageOps.invert(blur_mask)
				temp = stretching_image.copy()
				temp = temp.filter(blur)
				temp.paste(stretching_image, (0, 0), mask=blur_mask)
				img2img = {
					'steps': steps,
					'cfg_scale': cfg_scale,
					'sampler_name': sampler_name,
					'scheduler': scheduler,
					'denoise': denoise,
					'padding': 32,
					'dilation': dilation,
					'width': stretching_image.width,
					'height': stretching_image.height,
				}
				image = process.process_img2img_with_mask(bind, stretching_image, img2img, mask)
				stretching_image.save('stretching_image.png')
				mask.save('mask.png')
				results.append(image.convert('RGB'))
			elif method == 'inpaint+lama':
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio), dilation)
				lama = LamaInpainting()
				stretching_image = lama(stretching_image, mask)
				img2img = {
					'steps': steps,
					'cfg_scale': cfg_scale,
					'sampler_name': sampler_name,
					'scheduler': scheduler,
					'denoise': denoise,
					'padding': 32,
					'dilation': dilation,
					'width': stretching_image.width,
					'height': stretching_image.height,
				}
				image = process.process_img2img_with_mask(bind, stretching_image, img2img, mask)
				results.append(image.convert('RGB'))

		bind.pixels = utils.get_pixels_from_pils(results)
		return (bind, bind.pixels,)


class BMABResizeAndFill:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'width': ('INT', {'default': 1024, 'min': 0, 'max': 10000}),
				'height': ('INT', {'default': 1024, 'min': 0, 'max': 10000}),
				'fill_black': (('disable', 'enable'), )
			},
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/resize'

	def process(self, image, width, height, fill_black):
		results = []
		fill_black = fill_black == 'enable'
		for img in utils.get_pils_from_pixels(image):
			results.append(utils.resize_and_fill(img, width, height, fill_black=fill_black))
		pixels = utils.get_pixels_from_pils(results)
		return (pixels,)
