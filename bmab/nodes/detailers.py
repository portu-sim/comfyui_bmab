import comfy
import nodes
import math
import numpy as np
from collections.abc import Iterable

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from bmab import utils
from bmab.utils import yolo, sam
from bmab.nodes.binder import BMABBind

import folder_paths


class BMABDetailer:

	def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
		print(f'Loading lora {lora_name}')
		lora_path = folder_paths.get_full_path('loras', lora_name)
		lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
		model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
		return (model_lora, clip_lora)

	@staticmethod
	def apply_color_correction(correction, original_image):
		import cv2
		import numpy as np
		from skimage import exposure
		from blendmodes.blend import blendLayers, BlendType

		image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
			cv2.cvtColor(
				np.asarray(original_image),
				cv2.COLOR_RGB2LAB
			),
			cv2.cvtColor(np.asarray(correction.copy()), cv2.COLOR_RGB2LAB),
			channel_axis=2
		), cv2.COLOR_LAB2RGB).astype("uint8"))

		image = blendLayers(image, original_image, BlendType.LUMINOSITY)

		return image.convert('RGB')


class BMABFaceDetailer(BMABDetailer):
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 0.4, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'padding': ('INT', {'default': 32, 'min': 8, 'max': 128, 'step': 8}),
				'dilation': ('INT', {'default': 4, 'min': 4, 'max': 32, 'step': 1}),
				'width': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
			},
			'optional': {
				'image': ('IMAGE',),
				'lora': ('BMAB lora',)
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE')
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/detailer'

	def detailer(self, face, bind: BMABBind, steps, cfg, sampler_name, scheduler, denoise):
		pixels = utils.pil2tensor(face.convert('RGB'))
		latent = dict(samples=bind.vae.encode(pixels))
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise)[0]
		latent = bind.vae.decode(samples["samples"])
		return utils.tensor2pil(latent)

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, padding, dilation, width, height, image=None, lora=None):
		pixels = bind.pixels if image is None else image

		results = []
		for bgimg in utils.get_pils_from_pixels(pixels):
			if bind.context is not None:
				steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

			if lora is not None:
				for l in lora.loras:
					bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)

			boxes, confs = utils.yolo.predict(bgimg, 'face_yolov8n.pt', 0.35)
			for box, conf in zip(boxes, confs):
				x1, y1, x2, y2 = tuple(int(x) for x in box)
				x1, y1, x2, y2 = x1 - dilation, y1 - dilation, x2 + dilation, y2 + dilation
				cbx = utils.get_box_with_padding(bgimg, (x1, y1, x2, y2), padding)
				cropped = bgimg.crop(cbx)
				resized = utils.resize_and_fill(cropped, width, height)
				face = self.detailer(resized, bind, steps, cfg_scale, sampler_name, scheduler, denoise)
				face = self.apply_color_correction(resized, face)

				iratio = width / height
				cratio = cropped.width / cropped.height
				if iratio < cratio:
					ratio = cropped.width / width
					face = face.resize((int(face.width * ratio), int(face.height * ratio)))
					y0 = (face.height - cropped.height) // 2
					face = face.crop((0, y0, cropped.width, y0 + cropped.height))
				else:
					ratio = cropped.height / height
					face = face.resize((int(face.width * ratio), int(face.height * ratio)))
					x0 = (face.width - cropped.width) // 2
					face = face.crop((x0, 0, x0 + cropped.width, cropped.height))

				mask = Image.new('L', bgimg.size, 0)
				dr = ImageDraw.Draw(mask, 'L')
				dr.rectangle((x1, y1, x2, y2), fill=255)
				blur = ImageFilter.GaussianBlur(10)
				mask = mask.filter(blur)

				img = bgimg.copy()
				img.paste(face, (cbx[0], cbx[1]))
				bgimg.paste(img, (0, 0), mask=mask)

			results.append(bgimg.convert('RGB'))

		bind.pixels = utils.get_pixels_from_pils(results)
		return (bind, bind.pixels)


class BMABPersonDetailer(BMABDetailer):
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 0.4, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'upscale_ratio': ('FLOAT', {'default': 4.0, 'min': 1.0, 'max': 8.0, 'step': 0.01}),
				'dilation_mask': ('INT', {'default': 3, 'min': 3, 'max': 20, 'step': 1}),
				'large_person_area_limit': ('FLOAT', {'default': 0.1, 'min': 0.01, 'max': 1.0, 'step': 0.01}),
				'limit': ('INT', {'default': 1, 'min': 0, 'max': 20, 'step': 1}),
			},
			'optional': {
				'image': ('IMAGE',),
				'lora': ('BMAB lora',)
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE')
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/detailer'

	def process_img2img(self, image, bind: BMABBind, steps, cfg, sampler_name, scheduler, denoise):
		pixels = utils.pil2tensor(image.convert('RGB'))
		latent = dict(samples=bind.vae.encode(pixels))
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise)[0]
		latent = bind.vae.decode(samples["samples"])
		result = utils.tensor2pil(latent)
		return result

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, upscale_ratio, dilation_mask, large_person_area_limit, limit, image=None, lora=None):
		pixels = bind.pixels if image is None else image

		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

		if lora is not None:
			for l in lora.loras:
				bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)

		results = []
		for image in utils.get_pils_from_pixels(pixels):

			boxes, confs = yolo.predict(image, 'person_yolov8m-seg.pt', 0.35)
			if len(boxes) == 0:
				bind.pixels = pixels
				return (bind, bind.pixels, )

			for idx, (box, conf) in enumerate(zip(boxes, confs)):
				x1, y1, x2, y2 = tuple(int(x) for x in box)

				if limit != 0 and idx >= limit:
					break

				cropped = image.crop(box=(x1, y1, x2, y2))

				area_person = cropped.width * cropped.height
				area_image = image.width * image.height
				ratio = area_person / area_image
				print('AREA', area_image, (cropped.width * cropped.height), ratio)
				if ratio > 1 and ratio >= large_person_area_limit:
					print(f'Person is too big to process. {ratio} >= {large_person_area_limit}.')
					continue

				block_overscaled_image = True
				auto_upscale = True

				scale = upscale_ratio
				w, h = utils.fix_size_by_scale(cropped.width, cropped.height, scale)
				print(f'Trying x{scale} ({cropped.width},{cropped.height}) -> ({w},{h})')
				if scale > 1 and block_overscaled_image:
					area_scaled = w * h
					if area_scaled > area_image:
						print(f'It is too large to process.')
						if not auto_upscale:
							continue
						print('AREA', area_image, (cropped.width * cropped.height))
						scale = math.sqrt(area_image / (cropped.width * cropped.height))
						w, h = utils.fix_size_by_scale(cropped.width, cropped.height, scale)
						print(f'Auto Scale x{scale} ({cropped.width},{cropped.height}) -> ({w},{h})')
						if scale < 1.2:
							print(f'Scale {scale} has no effect. skip!!!!!')
							continue

				print(f'Scale {scale}')
				mask = sam.sam_predict_box(image, box).convert('L')

				enlarged = cropped.resize((w, h), Image.Resampling.LANCZOS)
				processed = self.process_img2img(enlarged, bind, steps, cfg_scale, sampler_name, scheduler, denoise)
				processed = processed.resize(cropped.size, Image.Resampling.LANCZOS)

				cropped_mask = mask.crop((x1, y1, x2, y2))
				blur = ImageFilter.GaussianBlur(4)
				blur_mask = cropped_mask.filter(blur)

				image.paste(processed, (x1, y1), mask=blur_mask)
			results.append(image)

		bind.pixels = utils.get_pixels_from_pils(results)
		return (bind, bind.pixels)


class BMABSimpleHandDetailer(BMABDetailer):
	@classmethod
	def INPUT_TYPES(s):
		try:
			from bmab.utils import grdino
			return {
				'required': {
					'bind': ('BMAB bind',),
					'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
					'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
					'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
					'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
					'denoise': ('FLOAT', {'default': 0.45, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
					'padding': ('INT', {'default': 32, 'min': 8, 'max': 128, 'step': 8}),
					'dilation': ('INT', {'default': 4, 'min': 4, 'max': 32, 'step': 1}),
					'width': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
					'height': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
				},
				'optional': {
					'image': ('IMAGE',),
					'lora': ('BMAB lora',)
				}
			}
		except:
			pass

		return {
			'required': {
				'text': (
					'STRING',
					{
						'default': 'Cannot Load GroundingDINO. To use this node, install GroudingDINO first.',
						'multiline': True,
						'dynamicPrompts': True
					}
				),
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE', 'IMAGE')
	RETURN_NAMES = ('BMAB bind', 'image', 'annotation')
	FUNCTION = 'process'

	CATEGORY = 'BMAB/detailer'

	def detailer(self, pil_img, bind: BMABBind, steps, cfg, sampler_name, scheduler, denoise):
		pixels = utils.pil2tensor(pil_img.convert('RGB'))
		latent = dict(samples=bind.vae.encode(pixels))
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise)[0]
		latent = bind.vae.decode(samples["samples"])
		return utils.tensor2pil(latent)

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, padding, dilation, width, height, image=None, lora=None):
		try:
			from bmab.utils import grdino
		except:
			print('-'*30)
			print('You should install GroudingDINO on your system.')
			print('-'*30)
			raise

		pixels = bind.pixels if image is None else image

		results = []
		bbox_results = []
		for bgimg in utils.get_pils_from_pixels(pixels):

			bounding_box = bgimg.convert('RGB').copy()
			bonding_dr = ImageDraw.Draw(bounding_box)

			if bind.context is not None:
				steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

			if lora is not None:
				for l in lora.loras:
					bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)

			boxes, logits, phrases = grdino.dino_predict(bgimg, 'hand', 0.35, 0.25)
			for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
				print(phrase)
				if phrase != 'hand':
					continue

				x1, y1, x2, y2 = tuple(int(x) for x in box)
				print('HAND', idx, (x1, y1, x2, y2))
				bonding_dr.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)

				x1, y1, x2, y2 = x1 - dilation, y1 - dilation, x2 + dilation, y2 + dilation
				cbx = x1 - padding, y1 - padding, x2 + padding, y2 + padding
				cropped = bgimg.crop(cbx)
				resized = utils.resize_and_fill(cropped, width, height)
				face = self.detailer(resized, bind, steps, cfg_scale, sampler_name, scheduler, denoise)

				iratio = width / height
				cratio = cropped.width / cropped.height
				if iratio < cratio:
					ratio = cropped.width / width
					face = face.resize((int(face.width * ratio), int(face.height * ratio)))
					y0 = (face.height - cropped.height) // 2
					face = face.crop((0, y0, cropped.width, y0 + cropped.height))
				else:
					ratio = cropped.height / height
					face = face.resize((int(face.width * ratio), int(face.height * ratio)))
					x0 = (face.width - cropped.width) // 2
					face = face.crop((x0, 0, x0 + cropped.width, cropped.height))

				mask = Image.new('L', bgimg.size, 0)
				dr = ImageDraw.Draw(mask, 'L')
				dr.rectangle((x1, y1, x2, y2), fill=255)
				blur = ImageFilter.GaussianBlur(4)
				mask = mask.filter(blur)

				img = bgimg.copy()
				img.paste(face, (cbx[0], cbx[1]))
				bgimg.paste(img, (0, 0), mask=mask)

			results.append(bgimg.convert('RGB'))
			bbox_results.append(bounding_box.convert('RGB'))

		bind.pixels = utils.get_pixels_from_pils(results)
		bounding_box = utils.get_pixels_from_pils(bbox_results)
		return (bind, bind.pixels, bounding_box)


class BMABSubframeHandDetailer(BMABDetailer):
	@classmethod
	def INPUT_TYPES(s):
		try:
			from bmab.utils import grdino
			return {
				'required': {
					'bind': ('BMAB bind',),
					'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
					'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
					'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
					'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
					'denoise': ('FLOAT', {'default': 0.45, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
					'padding': ('INT', {'default': 32, 'min': 8, 'max': 128, 'step': 8}),
					'dilation': ('INT', {'default': 4, 'min': 4, 'max': 32, 'step': 1}),
					'width': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
					'height': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
				},
				'optional': {
					'image': ('IMAGE',),
					'lora': ('BMAB lora',)
				}
			}
		except:
			pass

		return {
			'required': {
				'text': (
					'STRING',
					{
						'default': 'Cannot Load GroundingDINO. To use this node, install GroudingDINO first.',
						'multiline': True,
						'dynamicPrompts': True
					}
				),
			}
		}


	RETURN_TYPES = ('BMAB bind', 'IMAGE', 'IMAGE')
	RETURN_NAMES = ('BMAB bind', 'image', 'annotation')
	FUNCTION = 'process'

	CATEGORY = 'BMAB/detailer'

	class Obj(object):
		name = None

		def __init__(self, xyxy) -> None:
			super().__init__()
			self.parent = None
			self.xyxy = xyxy
			self.objects = []
			self.inbox = xyxy

		def is_in(self, obj) -> bool:
			x1, y1, x2, y2 = self.inbox
			mx1, my1, mx2, my2 = obj.xyxy

			x = int(x1 + (x2 - x1) / 2)
			y = int(y1 + (y2 - y1) / 2)

			return mx1 <= x <= mx2 and my1 <= y <= my2

		def append(self, obj):
			obj.parent = self
			for ch in self.objects:
				if obj.is_in(ch):
					obj.parent = ch
					break
			self.objects.append(obj)

		def is_valid(self):
			return True

		def size(self):
			x1, y1, x2, y2 = self.xyxy
			return (x2 - x1) * (y2 - y1)

		def put(self, mask):
			for xg in self.objects:
				if not xg.is_valid():
					continue
				if xg.name == 'hand':
					dr = ImageDraw.Draw(mask, 'L')
					dr.rectangle(xg.xyxy, fill=255)

		def get_box(self):
			if not self.objects:
				return self.xyxy

			x1, y1, x2, y2 = self.xyxy
			ret = [x2, y2, x1, y1]
			for xg in self.objects:
				if not xg.is_valid():
					continue
				x = xg.xyxy
				ret[0] = x[0] if x[0] < ret[0] else ret[0]
				ret[1] = x[1] if x[1] < ret[1] else ret[1]
				ret[2] = x[2] if x[2] > ret[2] else ret[2]
				ret[3] = x[3] if x[3] > ret[3] else ret[3]

			return x1, y1, x2, ret[3]

		def log(self):
			print(self.name, self.xyxy)
			for x in self.objects:
				x.log()

	class Person(Obj):
		name = 'person'

		def __init__(self, xyxy, scale) -> None:
			super().__init__(xyxy)
			self.inbox = utils.fix_box_by_scale(xyxy, scale)

		def is_valid(self):
			face = False
			hand = False
			for xg in self.objects:
				if xg.name == 'face':
					face = True
				if xg.name == 'hand':
					hand = True
			return face and hand

		def cleanup(self):
			print([xg.name for xg in self.objects])
			nw = []
			for xg in self.objects:
				if xg.name == 'person':
					if len(self.objects) == 1 and xg.is_valid():
						self.xyxy = xg.xyxy
						self.objects = xg.objects
						return
					else:
						self.objects.extend(xg.objects)
				else:
					nw.append(xg)
			self.objects = nw

	class Head(Obj):
		name = 'head'

	class Face(Obj):
		name = 'face'

	class Hand(Obj):
		name = 'hand'

	def get_subframe(self, pilimg, scale, box_threshold=0.30, text_threshold=0.20):
		text_prompt = "person . head . face . hand ."
		print('threshold', box_threshold)
		from bmab.utils import grdino

		boxes, logits, phrases = grdino.dino_predict(pilimg, text_prompt, box_threshold, text_threshold)

		people = []

		def find_person(o):
			for person in people:
				if o.is_in(person):
					return person
			return None

		for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
			if phrase == 'person':
				p = BMABSubframeHandDetailer.Person(tuple(int(x) for x in box), scale)
				parent = find_person(p)
				if parent:
					parent.append(p)
				else:
					people.append(p)
		people = sorted(people, key=lambda c: c.size(), reverse=True)

		for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
			bb = tuple(int(x) for x in box)
			print(float(logit), phrase, bb)

			if phrase == 'head':
				o = BMABSubframeHandDetailer.Head(bb)
				parent = find_person(o)
				if parent:
					parent.append(o)
			elif phrase == 'face' or phrase == 'head face':
				o = BMABSubframeHandDetailer.Face(bb)
				parent = find_person(o)
				if parent:
					parent.append(o)
			elif phrase == 'hand':
				o = BMABSubframeHandDetailer.Hand(bb)
				parent = find_person(o)
				if parent:
					parent.append(o)

		for person in people:
			person.cleanup()

		boxes = []
		masks = []
		for person in people:
			if person.is_valid():
				mask = Image.new('L', pilimg.size, color=0)
				person.log()
				person.put(mask)
				boxes.append(person.get_box())
				masks.append(mask)
		return boxes, masks

	def detailer(self, pil_img, bind: BMABBind, steps, cfg, sampler_name, scheduler, denoise):
		pixels = utils.pil2tensor(pil_img.convert('RGB'))
		latent = dict(samples=bind.vae.encode(pixels))
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise)[0]
		latent = bind.vae.decode(samples["samples"])
		return utils.tensor2pil(latent)

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, padding, dilation, width, height, image=None, lora=None):
		try:
			from bmab.utils import grdino
		except:
			print('-'*30)
			print('You should install GroudingDINO on your system.')
			print('-'*30)
			raise

		pixels = bind.pixels if image is None else image

		results = []
		bbox_results = []
		for bgimg in utils.get_pils_from_pixels(pixels):

			bounding_box = bgimg.convert('RGB').copy()
			bonding_dr = ImageDraw.Draw(bounding_box)

			if bind.context is not None:
				steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

			boxes, masks = self.get_subframe(bgimg, 0, box_threshold=0.35)
			if len(boxes) > 0:
				if lora is not None:
					for l in lora.loras:
						bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)

				for box, mask in zip(boxes, masks):
					box = utils.fix_box_by_scale(box, 0)
					box = utils.fix_box_size(box)
					box = utils.fix_box_limit(box, bgimg.size)
					x1, y1, x2, y2 = box

					x1, y1, x2, y2 = x1 - dilation, y1 - dilation, x2 + dilation, y2 + dilation

					bonding_dr.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)

					cbx = x1 - padding, y1 - padding, x2 + padding, y2 + padding

					cropped = bgimg.crop(cbx)
					resized = utils.resize_and_fill(cropped, width, height)
					subframe = self.detailer(resized, bind, steps, cfg_scale, sampler_name, scheduler, denoise)

					iratio = width / height
					cratio = cropped.width / cropped.height
					if iratio < cratio:
						ratio = cropped.width / width
						subframe = subframe.resize((int(subframe.width * ratio), int(subframe.height * ratio)))
						y0 = (subframe.height - cropped.height) // 2
						subframe = subframe.crop((0, y0, cropped.width, y0 + cropped.height))
					else:
						ratio = cropped.height / height
						subframe = subframe.resize((int(subframe.width * ratio), int(subframe.height * ratio)))
						x0 = (subframe.width - cropped.width) // 2
						subframe = subframe.crop((x0, 0, x0 + cropped.width, cropped.height))

					blur = ImageFilter.GaussianBlur(4)
					mask = mask.filter(blur)

					img = bgimg.copy()
					img.paste(subframe, (cbx[0], cbx[1]))
					bgimg.paste(img, (0, 0), mask=mask)
			results.append(bgimg)
			bbox_results.append(bounding_box)

		bind.pixels = utils.get_pixels_from_pils(results)
		bounding_box = utils.get_pixels_from_pils(bbox_results)
		return (bind, bind.pixels, bounding_box)


class BMABDetailAnything(BMABDetailer):

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'masks': ('MASK',),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 0.4, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'padding': ('INT', {'default': 32, 'min': 8, 'max': 128, 'step': 8}),
				'dilation': ('INT', {'default': 4, 'min': 4, 'max': 32, 'step': 1}),
				'width': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
				'limit': ('INT', {'default': 1, 'min': 0, 'max': 20, 'step': 1}),
			},
			'optional': {
				'image': ('IMAGE',),
				'lora': ('BMAB lora',)
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE')
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/detailer'

	def process_img2img(self, image, bind: BMABBind, steps, cfg, sampler_name, scheduler, denoise):
		pixels = utils.pil2tensor(image.convert('RGB'))
		latent = dict(samples=bind.vae.encode(pixels))
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise)[0]
		latent = bind.vae.decode(samples["samples"])
		result = utils.tensor2pil(latent)
		return result

	def process(self, bind: BMABBind, masks, steps, cfg_scale, sampler_name, scheduler, denoise, padding, dilation, width, height, limit, image=None, lora=None):
		pixels = bind.pixels if image is None else image

		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

		if lora is not None:
			for l in lora.loras:
				bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)

		images = utils.get_pils_from_pixels(pixels)

		results = []
		for image, _masks in zip(images, masks):
			for (idx, m) in enumerate(_masks):
				i = 255. * m.cpu().numpy().squeeze()
				pil_mask = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
				box = pil_mask.getbbox()
				x1, y1, x2, y2 = tuple(int(x) for x in box)

				if limit != 0 and idx >= limit:
					break

				cbx = utils.get_box_with_padding(image, (x1, y1, x2, y2), padding)
				cropped = image.crop(cbx)
				resized = utils.resize_and_fill(cropped, width, height)
				processed = self.process_img2img(resized, bind, steps, cfg_scale, sampler_name, scheduler, denoise)
				processed = self.apply_color_correction(resized, processed)

				iratio = width / height
				cratio = cropped.width / cropped.height
				if iratio < cratio:
					ratio = cropped.width / width
					processed = processed.resize((int(processed.width * ratio), int(processed.height * ratio)))
					y0 = (processed.height - cropped.height) // 2
					processed = processed.crop((0, y0, cropped.width, y0 + cropped.height))
				else:
					ratio = cropped.height / height
					processed = processed.resize((int(processed.width * ratio), int(processed.height * ratio)))
					x0 = (processed.width - cropped.width) // 2
					processed = processed.crop((x0, 0, x0 + cropped.width, cropped.height))

				img = image.copy()
				img.paste(processed, (cbx[0], cbx[1]))

				pil_mask = utils.dilate_mask(pil_mask, dilation)
				blur = ImageFilter.GaussianBlur(dilation)
				blur_mask = pil_mask.filter(blur)

				image.paste(img, (0, 0), mask=blur_mask)
			results.append(image)

		bind.pixels = utils.get_pixels_from_pils(results)
		return (bind, bind.pixels)



