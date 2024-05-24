import comfy
import nodes
import math
import numpy as np
from collections.abc import Iterable

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

from bmab import utils
from bmab import process
from bmab.utils import yolo, sam
from bmab.nodes.binder import BMABBind
from bmab.nodes.cnloader import BMABControlNetOpenpose

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
	models = ['bmab_face_nm_yolov8n.pt', 'bmab_face_sm_yolov8n.pt', 'face_yolov8n.pt', 'face_yolov8m.pt', 'face_yolov8n_v2.pt', 'face_yolov8s.pt']
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
				'model': (s.models, ),
				'limit': ('INT', {'default': 1, 'min': 0, 'max': 5, 'step': 1}),
				'order': (['Size', 'Left', 'Right', 'Center', 'Score'], ),
			},
			'optional': {
				'image': ('IMAGE',),
				'lora': ('BMAB lora',)
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE')
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/detailer'

	def detailer(self, face, bind: BMABBind, steps, cfg, sampler_name, scheduler, denoise):
		pixels = utils.pil2tensor(face.convert('RGB'))
		latent = dict(samples=bind.vae.encode(pixels))
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise)[0]
		latent = bind.vae.decode(samples["samples"])
		return utils.tensor2pil(latent)

	def process_faces(self, bind, bgimg, img2img, model, limit, order):
		boxes, logits = utils.yolo.predict(bgimg, model, 0.35)
		candidate = []
		if order == 'Left':
			for box, logit in zip(boxes, logits):
				x1, y1, x2, y2 = box
				value = x1 + (x2 - x1) // 2
				print('detected(from left)', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0])
		elif order == 'Right':
			for box, logit in zip(boxes, logits):
				x1, y1, x2, y2 = box
				value = x1 + (x2 - x1) // 2
				print('detected(from right)', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		elif order == 'Center':
			for box, logit in zip(boxes, logits):
				x1, y1, x2, y2 = box
				cx = bgimg.width / 2
				cy = bgimg.height / 2
				ix = x1 + (x2 - x1) // 2
				iy = y1 + (y2 - y1) // 2
				value = math.sqrt(abs(cx - ix) ** 2 + abs(cy - iy) ** 2)
				print('detected(from center)', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0])
		elif order == 'Size':
			for box, logit in zip(boxes, logits):
				x1, y1, x2, y2 = box
				value = (x2 - x1) * (y2 - y1)
				print('detected(size)', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)
		else:
			for box, logit in zip(boxes, logits):
				value = float(logit)
				print(f'detected({order})', float(logit), value)
				candidate.append((value, box, logit))
			candidate = sorted(candidate, key=lambda c: c[0], reverse=True)

		for index, (value, box, logit) in enumerate(candidate):
			if limit != 0 and index > limit:
				return bgimg
			bgimg = process.process_img2img_with_mask(bind, bgimg, img2img, box=box)
		return bgimg

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, padding, dilation, width, height, model, limit, order, image=None, lora=None):
		pixels = bind.pixels if image is None else image

		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

		if lora is not None:
			for l in lora.loras:
				bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)

		results = []
		for bgimg in utils.get_pils_from_pixels(pixels):
			img2img = {
				'steps': steps,
				'cfg_scale': cfg_scale,
				'sampler_name': sampler_name,
				'scheduler': scheduler,
				'denoise': denoise,
				'padding': padding,
				'dilation': dilation,
				'width': width,
				'height': height,
			}
			results.append(self.process_faces(bind, bgimg, img2img, model, limit, order))

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
	RETURN_NAMES = ('BMAB bind', 'image',)
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
				return (bind, bind.pixels,)

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
			print('-' * 30)
			print('You should install GroudingDINO on your system.')
			print('-' * 30)
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
					'squeeze': (('disable', 'enable'),),
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

	def process_person(self, bind: BMABBind, bgimg: Image, person, person_hand, squeeze, img2img):
		width, height, padding, dilation = img2img['width'], img2img['height'], img2img['padding'], img2img['dilation']

		x1, y1, x2, y2 = tuple(int(x) for x in person)
		l_y = [hand[3] for hand in person_hand]
		max_y = max(l_y)
		box = (x1, y1, x2, max_y)
		if max_y < y1:
			return bgimg, (x1, y1, x2, y2)

		if squeeze:
			cr = bgimg.crop((x1, y1, x2, max_y))
			bx, _ = utils.yolo.predict(cr, 'person_yolov8m-seg.pt', 0.35)
			if len(bx) > 0:
				bs = bx[0]
				x, y = box[0], box[1]
				box = bs[0] + x, bs[1] + y, bs[2] + x, bs[3] + y
				m = sam.sam_predict_box(bgimg, box).convert('L')
				box = m.getbbox()

		cbx = utils.get_box_with_padding(bgimg, box, padding)

		# Process Img2img
		processed = process.process_img2img_with_mask(bind, bgimg.copy(), img2img, box=cbx)

		# Paste hand into org image
		mask = Image.new('L', bgimg.size, 0)
		dr = ImageDraw.Draw(mask, 'L')
		for hand in person_hand:
			dr.rectangle(hand, fill=255)
		pil_mask = utils.dilate_mask(mask, dilation)
		blur = ImageFilter.GaussianBlur(dilation)
		blur_mask = pil_mask.filter(blur)
		bgimg.paste(processed, (0, 0), mask=blur_mask)

		return bgimg, cbx

	def process_image(self, bind: BMABBind, bgimg: Image, squeeze, img2img: dict):

		text_prompt = "person . head . face . hand ."
		from bmab.utils import grdino

		hand_boxes = []
		boxes, logits, phrases = grdino.dino_predict(bgimg, text_prompt, 0.35, 0.25)
		persons = [tuple(int(x) for x in box) for box, phrase in zip(boxes, phrases) if phrase == 'person']
		hands = [tuple(int(x) for x in box) for box, phrase in zip(boxes, phrases) if phrase == 'hand']

		print('Person', len(persons))
		print('Hand', len(hands))

		bounding_box = bgimg.convert('RGB').copy()

		person_bouding_box = []
		for person in persons:
			person_hand = [hand for hand in hands if utils.is_box_in_box(hand, person)]
			if len(person_hand) == 0:
				continue
			hand_boxes.extend(person_hand)

			bgimg, cbx = self.process_person(bind, bgimg, person, person_hand, squeeze, img2img)
			person_bouding_box.append(cbx)

		bonding_dr = ImageDraw.Draw(bounding_box)
		for person_box in person_bouding_box:
			bonding_dr.rectangle(person_box, outline=(0, 255, 0), width=2)
		for hand_bbox in hand_boxes:
			bonding_dr.rectangle(hand_bbox, outline=(255, 0, 0), width=2)

		return bgimg, bounding_box

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, padding, dilation, width, height, squeeze, image=None, lora=None):
		try:
			from bmab.utils import grdino
		except:
			print('-' * 30)
			print('You should install GroudingDINO on your system.')
			print('-' * 30)
			raise

		squeeze = squeeze == 'enable'
		pixels = bind.pixels if image is None else image

		if lora is not None:
			for l in lora.loras:
				bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)

		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

		img2img = {
			'steps': steps,
			'cfg_scale': cfg_scale,
			'sampler_name': sampler_name,
			'scheduler': scheduler,
			'denoise': denoise,
			'padding': padding,
			'dilation': dilation,
			'width': width,
			'height': height,
		}

		results = []
		bbox_results = []
		for bgimg in utils.get_pils_from_pixels(pixels):
			bgimg = bgimg.convert('RGB')
			result, bbox_result = self.process_image(bind, bgimg, squeeze, img2img)
			results.append(result)
			bbox_results.append(bbox_result)

		bind.pixels = utils.get_pixels_from_pils(results)
		bounding_box = utils.get_pixels_from_pils(bbox_results)

		return (bind, bind.pixels, bounding_box,)


class BMABOpenposeHandDetailer(BMABDetailer):
	@classmethod
	def INPUT_TYPES(s):
		try:
			from bmab.utils import grdino
			from comfyui_controlnet_aux.node_wrappers.dwpose import DWPose_Preprocessor
			from comfyui_controlnet_aux.node_wrappers.openpose import OpenPose_Preprocessor

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
					'squeeze': (('disable', 'enable'),),
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

	def get_pose(self, image):
		from comfyui_controlnet_aux.node_wrappers import dwpose
		from controlnet_aux.open_pose import OpenposeDetector
		from comfy import model_management

		bbox_detector = "yolox_l.onnx"
		pose_estimator = "dw-ll_ucoco_384.onnx"

		model = dwpose.DwposeDetector.from_pretrained(
			dwpose.DWPOSE_MODEL_NAME,
			dwpose.DWPOSE_MODEL_NAME,
			det_filename=bbox_detector, pose_filename=pose_estimator,
			torchscript_device=model_management.get_torch_device()
		)
		self.openpose_dicts = []

		def func(image, **kwargs):
			pose_img, openpose_dict = model(image, **kwargs)
			self.openpose_dicts.append(openpose_dict)
			return pose_img

		out = dwpose.common_annotator_call(func, image, include_hand=True, include_face=True, include_body=True, image_and_json=True, resolution=1024)
		del model
		return out

	def process_openpose(self, image):
		tensor = utils.pil2tensor(image)
		out = self.get_pose(tensor)
		pose_image = utils.tensor2pil(out)
		pose_image = pose_image.resize(image.size, Image.Resampling.LANCZOS)
		return pose_image

	def process_person(self, bind: BMABBind, bgimg: Image, bounding_box: Image, person, person_hand, squeeze, img2img):
		width, height, padding, dilation = img2img['width'], img2img['height'], img2img['padding'], img2img['dilation']

		x1, y1, x2, y2 = tuple(int(x) for x in person)
		l_y = [hand[3] for hand in person_hand]
		max_y = max(l_y)
		box = (x1, y1, x2, max_y)

		if squeeze:
			cr = bgimg.crop((x1, y1, x2, max_y))
			bx, _ = utils.yolo.predict(cr, 'person_yolov8m-seg.pt', 0.35)
			if len(bx) > 0:
				bs = bx[0]
				x, y = box[0], box[1]
				box = bs[0] + x, bs[1] + y, bs[2] + x, bs[3] + y
				m = sam.sam_predict_box(bgimg, box).convert('L')
				box = m.getbbox()

		cbx = utils.get_box_with_padding(bgimg, box, padding)

		cropped = bgimg.crop(cbx)
		resized = utils.resize_and_fill(cropped, width, height)
		posed = self.process_openpose(resized)

		# Prepare ControlNet
		bind_2 = bind.copy()
		openpose = BMABControlNetOpenpose()
		cnname = openpose.get_openpose_filenames()[0]
		pose_pixel = utils.pil2tensor(posed)
		openpose.apply_controlnet(bind_2, cnname, 1.0, 0, 1, None, image_in=pose_pixel)

		# Process Img2img
		processed = process.process_img2img_with_mask(bind, bgimg.copy(), img2img, box=cbx)

		# Paste hand into org image
		mask = Image.new('L', bgimg.size, 0)
		dr = ImageDraw.Draw(mask, 'L')
		for hand in person_hand:
			dr.rectangle(hand, fill=255)
		pil_mask = utils.dilate_mask(mask, dilation)
		blur = ImageFilter.GaussianBlur(dilation)
		blur_mask = pil_mask.filter(blur)
		bgimg.paste(processed, (0, 0), mask=blur_mask)

		## this is for annotation
		# Revert Pose
		posed = utils.revert_image(width, height, posed, cropped)

		# Overlay annotation image
		mdata = posed.convert('RGBA').getdata()
		newdata = []
		for idx in range(0, len(mdata)):
			if mdata[idx][0] == 0 and mdata[idx][1] == 0 and mdata[idx][2] == 0:
				newdata.append((0, 0, 0, 0))
			else:
				newdata.append(mdata[idx])
		posed_rgba = Image.new('RGBA', posed.size)
		posed_rgba.putdata(newdata)
		temp = Image.new('RGBA', bounding_box.size, (0, 0, 0, 0))
		temp.paste(posed_rgba, (cbx[0], cbx[1]))
		bounding_box = Image.alpha_composite(bounding_box.convert('RGBA'), temp).convert('RGB')

		return bgimg, bounding_box, cbx

	def process_image(self, bind: BMABBind, bgimg: Image, squeeze, img2img: dict):

		text_prompt = "person . head . face . hand ."
		from bmab.utils import grdino

		hand_boxes = []
		boxes, logits, phrases = grdino.dino_predict(bgimg, text_prompt, 0.35, 0.25)
		persons = [tuple(int(x) for x in box) for box, phrase in zip(boxes, phrases) if phrase == 'person']
		hands = [tuple(int(x) for x in box) for box, phrase in zip(boxes, phrases) if phrase == 'hand']

		print('Person', len(persons))
		print('Hand', len(hands))

		bounding_box = bgimg.convert('RGB').copy()

		person_bouding_box = []
		for person in persons:
			person_hand = [hand for hand in hands if utils.is_box_in_box(hand, person)]
			if len(person_hand) == 0:
				continue
			hand_boxes.extend(person_hand)

			bgimg, bounding_box, cbx = self.process_person(bind, bgimg, bounding_box, person, person_hand, squeeze, img2img)
			person_bouding_box.append(cbx)

		bonding_dr = ImageDraw.Draw(bounding_box)
		for person_box in person_bouding_box:
			bonding_dr.rectangle(person_box, outline=(0, 255, 0), width=2)
		for hand_bbox in hand_boxes:
			bonding_dr.rectangle(hand_bbox, outline=(255, 0, 0), width=2)

		return bgimg, bounding_box

	def process(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, padding, dilation, width, height, squeeze, image=None, lora=None):
		try:
			from bmab.utils import grdino
		except:
			print('-' * 30)
			print('You should install GroudingDINO on your system.')
			print('-' * 30)
			raise

		squeeze = squeeze == 'enable'
		pixels = bind.pixels if image is None else image

		if lora is not None:
			for l in lora.loras:
				bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)

		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)

		img2img = {
			'steps': steps,
			'cfg_scale': cfg_scale,
			'sampler_name': sampler_name,
			'scheduler': scheduler,
			'denoise': denoise,
			'padding': padding,
			'dilation': dilation,
			'width': width,
			'height': height,
		}

		results = []
		bbox_results = []
		for bgimg in utils.get_pils_from_pixels(pixels):
			bgimg = bgimg.convert('RGB')
			result, bbox_result = self.process_image(bind, bgimg, squeeze, img2img)
			results.append(result)
			bbox_results.append(bbox_result)

		bind.pixels = utils.get_pixels_from_pils(results)
		bounding_box = utils.get_pixels_from_pils(bbox_results)

		return (bind, bind.pixels, bounding_box,)


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
	RETURN_NAMES = ('BMAB bind', 'image',)
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
