import comfy
import folder_paths

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
				'ratio': ('FLOAT', {'default': 0.85, 'min': 0.1, 'max': 0.95, 'step': 0.01}),
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

			boxes, confs = predict(image, 'person_yolov8n-seg.pt', 0.35)
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
			print(f'Ratio {pratio:.2}, {ratio:.2}, {(pratio / ratio):.2}')
			if pratio > ratio:
				image_ratio = pratio / ratio
				if image_ratio < 1.0:
					results.append(image.convert('RGB'))
					continue
			else:
				results.append(image.convert('RGB'))
				continue

			print('Process image resize', image_ratio)

			stretching_image = utils.resize_image_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio))
			if method == 'stretching':
				results.append(stretching_image.convert('RGB'))
			elif method == 'inpaint':
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio))
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
				results.append(image.convert('RGB'))
			elif method == 'inpaint+lama':
				max_length = max(image.width, image.height)
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio))

				# lama image should be 512*512
				lama_img = utils.resize_and_fill(stretching_image, max_length, max_length).resize((512, 512), Image.Resampling.LANCZOS)
				lama_msk = utils.resize_and_fill(mask, max_length, max_length).resize((512, 512), Image.Resampling.LANCZOS)
				lama = LamaInpainting()
				lama_result = lama(lama_img, lama_msk)

				# resize back to streching image
				recovery_image = lama_result.resize((max_length, max_length), Image.Resampling.LANCZOS)
				blur = ImageFilter.GaussianBlur(4)
				blur_mask = mask.filter(blur)
				stretching_image.paste(recovery_image, (0, 0), mask=blur_mask)

				stretching_image.save('stretching_image.png')
				mask.save('mask.png')
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


import torch


class BMABResizeByRatio:
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
				'ratio': ('FLOAT', {'default': 0.85, 'min': 0.1, 'max': 0.95, 'step': 0.01}),
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

			image_ratio = 1 / ratio
			print('Process image resize', image_ratio)

			stretching_image = utils.resize_image_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio))
			if method == 'stretching':
				results.append(stretching_image.convert('RGB'))
			elif method == 'inpaint':
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio))
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
				results.append(image.convert('RGB'))
			elif method == 'inpaint+lama':
				max_length = max(image.width, image.height)
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * image_ratio), int(image.height * image_ratio))

				# lama image should be 512*512
				lama_img = utils.resize_and_fill(stretching_image, max_length, max_length).resize((512, 512), Image.Resampling.LANCZOS)
				lama_msk = utils.resize_and_fill(mask, max_length, max_length).resize((512, 512), Image.Resampling.LANCZOS)
				lama = LamaInpainting()
				lama_result = lama(lama_img, lama_msk)

				# resize back to streching image
				recovery_image = lama_result.resize((max_length, max_length), Image.Resampling.LANCZOS)
				recovery_image = utils.crop(recovery_image, image.width, image.height, False)

				blur = ImageFilter.GaussianBlur(4)
				blur_mask = mask.filter(blur)
				stretching_image.paste(recovery_image, (0, 0), mask=blur_mask)

				stretching_image.save('stretching_image.png')
				mask.save('mask.png')
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


class BMABCrop:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'image': ('IMAGE',),
				'width': ('INT', {'default': 2, 'min': 0, 'max': 10000}),
				'height': ('INT', {'default': 3, 'min': 0, 'max': 10000}),
				'resize': (('disable', 'enable'), )
			},
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/resize'

	def process(self, image, width, height, resize):
		results = []
		resize = resize == 'enable'
		for img in utils.get_pils_from_pixels(image):
			results.append(utils.crop(img, width, height, resized=resize))
		pixels = utils.get_pixels_from_pils(results)
		return (pixels,)


class BMABZoomOut:
	resize_methods = ['stretching', 'inpaint', 'inpaint+lama', 'controlnet']
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
				'ratio': ('FLOAT', {'default': 1.25, 'min': 1.0, 'max': 2.0, 'step': 0.01}),
				'dilation': ('INT', {'default': 30, 'min': 4, 'max': 128, 'step': 1}),
				'control_net_name': (folder_paths.get_filename_list('controlnet'),),
			},
			'optional': {
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE', )
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/resize'

	def generate(self, width, height, batch_size=1):
		self.device = comfy.model_management.intermediate_device()
		latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
		return latent

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, method, alignment, ratio, dilation, control_net_name, image=None):
		pixels = bind.pixels if image is None else image

		results = []
		for image in utils.get_pils_from_pixels(pixels):

			if bind.context is not None:
				steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

			print('Process image resize', ratio)

			stretching_image = utils.resize_image_with_alignment(image, alignment, int(image.width * ratio), int(image.height * ratio))
			if method == 'stretching':
				results.append(stretching_image.convert('RGB'))
			elif method == 'inpaint':
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * ratio), int(image.height * ratio))
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
				results.append(image.convert('RGB'))
			elif method == 'controlnet':
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * ratio), int(image.height * ratio))
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
				image = process.process_img2img_with_controlnet_mask(bind, control_net_name, stretching_image, img2img, mask)
				results.append(image.convert('RGB'))
			elif method == 'inpaint+lama':
				max_length = max(image.width, image.height)
				mask, box = utils.get_mask_with_alignment(image, alignment, int(image.width * ratio), int(image.height * ratio))

				# lama image should be 512*512
				lama_img = utils.resize_and_fill(stretching_image, max_length, max_length).resize((512, 512), Image.Resampling.LANCZOS)
				lama_msk = utils.resize_and_fill(mask, max_length, max_length).resize((512, 512), Image.Resampling.LANCZOS)
				lama = LamaInpainting()
				lama_result = lama(lama_img, lama_msk)

				# resize back to streching image
				recovery_image = lama_result.resize((max_length, max_length), Image.Resampling.LANCZOS)
				recovery_image = utils.crop(recovery_image, image.width, image.height, False)
				blur = ImageFilter.GaussianBlur(4)
				blur_mask = mask.filter(blur)
				stretching_image.paste(recovery_image, (0, 0), mask=blur_mask)

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


class BMABSquare:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'method': (['stretching', 'inpaint+lama'], ),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 1, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'dilation': ('INT', {'default': 32, 'min': 4, 'max': 128, 'step': 1}),
				'size': ('INT', {'default': 1024, 'min': 256, 'max': 4096, 'step': 8}),
				'control_net_name': (folder_paths.get_filename_list('controlnet'),),
			},
			'optional': {
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE', )
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/resize'

	def get_noise_pil(self, bind, size):
		device = comfy.model_management.intermediate_device()
		latent = torch.zeros([1, 4, size // 8, size // 8], device=device)
		pixels = bind.vae.decode(latent)
		return utils.get_pils_from_pixels(pixels).pop()

	def process(self, bind: BMABBind, method, steps, cfg_scale, sampler_name, scheduler, denoise, dilation, size, control_net_name, image=None):
		results = []
		for pil_img in utils.get_pils_from_pixels(image):
			w, h = pil_img.size
			if w == size and h == size:
				results.append(pil_img)
				continue

			if w == h:
				results.append(pil_img.resize((size, size), Image.Resampling.LANCZOS))
				continue

			img2img = {
				'steps': steps,
				'cfg_scale': cfg_scale,
				'sampler_name': sampler_name,
				'scheduler': scheduler,
				'denoise': denoise,
				'padding': 32,
				'dilation': dilation,
				'width': size,
				'height': size,
			}

			if method == 'inpaint+lama':
				pil_img = self.process_lama(bind, control_net_name, w, h, size, img2img, pil_img)
				results.append(pil_img.convert('RGB'))
			else:
				img = pil_img.resize((size, size), Image.Resampling.LANCZOS).convert('RGBA')
				rgba = Image.new('RGBA', (size, size), color=0)
				if w > h:
					if w != size:
						pil_img = pil_img.resize((size, int(h * (size / w))), Image.Resampling.LANCZOS)
					w, h = pil_img.size
					delta = (size - h) // 2
					mask = Image.new('L', (size, size), 0)
					dr = ImageDraw.Draw(mask)
					delta_dila = delta + img2img['dilation']
					dr.rectangle((0, 0, size, delta_dila), 255)
					dr.rectangle((0, size - delta_dila, size, size), 255)
					p_box = (0, delta)
				else:
					if h != size:
						pil_img = pil_img.resize((int(w * (size / h)), size), Image.Resampling.LANCZOS)
					w, h = pil_img.size
					delta = (size - w) // 2
					mask = Image.new('L', (size, size), 255)
					dr = ImageDraw.Draw(mask)
					delta_dila = delta + img2img['dilation']
					dr.rectangle((0, 0, delta_dila, size), 0)
					dr.rectangle((size - delta_dila, 0, size, size), 0)
					p_box = (delta, 0)

				blur = ImageFilter.GaussianBlur(img2img['dilation'] // 2)
				blur_mask = mask.filter(blur)
				rgba.paste(img, (0, 0))
				rgba.paste(pil_img, p_box)
				r, g, b, a = rgba.split()
				rgba = Image.merge('RGBA', (r, g, b, blur_mask))
				img = Image.alpha_composite(img, rgba)

				lama_img = img.resize((512, 512), Image.Resampling.LANCZOS)
				lama_msk = ImageOps.invert(mask).resize((512, 512), Image.Resampling.LANCZOS)
				lama = LamaInpainting()
				lama_result = lama(lama_img, lama_msk)
				# resize back to streching image
				recovery_image = lama_result.resize((size, size), Image.Resampling.LANCZOS)
				processed = process.process_img2img(bind, recovery_image, img2img)
				processed = processed.convert('RGBA')

				img = Image.alpha_composite(processed, rgba)

				results.append(img.convert('RGB'))

		pixels = utils.get_pixels_from_pils(results)
		return (bind, pixels, )

	def process_lama(self, bind, control_net_name, w, h, size, img2img, pil_img):
		if w > h:
			if w != size:
				pil_img = pil_img.resize((size, int(h * (size / w))), Image.Resampling.LANCZOS)
			w, h = pil_img.size
			delta = (size - h) // 2
			stretching_image = utils.resize_and_fill(pil_img, size, size)
			mask = Image.new('L', (size, size), 0)
			dr = ImageDraw.Draw(mask)
			delta_dila = delta + img2img['dilation']
			dr.rectangle((0, 0, size, delta_dila), 255)
			dr.rectangle((0, size - delta_dila, size, size), 255)
		else:
			if h != size:
				pil_img = pil_img.resize((int(w * (size / h)), size), Image.Resampling.LANCZOS)
			w, h = pil_img.size

			delta = (size - w) // 2
			stretching_image = utils.resize_and_fill(pil_img, size, size)
			mask = Image.new('L', (size, size), 0)
			dr = ImageDraw.Draw(mask)
			delta_dila = delta + img2img['dilation']
			dr.rectangle((0, 0, delta_dila, size), 255)
			dr.rectangle((size - delta_dila, 0, size, size), 255)
		# lama image should be 512*512
		lama_img = utils.resize_and_fill(stretching_image, size, size).resize((512, 512), Image.Resampling.LANCZOS)
		lama_msk = utils.resize_and_fill(mask, size, size).resize((512, 512), Image.Resampling.LANCZOS)
		lama = LamaInpainting()
		lama_result = lama(lama_img, lama_msk)
		# resize back to streching image
		recovery_image = lama_result.resize((size, size), Image.Resampling.LANCZOS)
		blur = ImageFilter.GaussianBlur(4)
		blur_mask = mask.filter(blur)
		stretching_image.paste(recovery_image, (0, 0), mask=blur_mask)
		pil_img = process.process_img2img_with_controlnet_mask(bind, control_net_name, stretching_image, img2img, mask)
		return pil_img



