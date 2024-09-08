import cv2
import comfy
import torch
import numpy as np
import folder_paths
import node_helpers
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageSequence
from bmab import utils
from bmab.nodes.binder import BMABBind
from bmab.external.rmbg14.briarmbg import BriaRMBG
from bmab.external.rmbg14.utilities import preprocess_image, postprocess_image


class BMABControlNet:

	@classmethod
	def INPUT_TYPES(s):
		input_dir = folder_paths.get_input_directory()
		files = ['None']
		files.extend(utils.get_file_list(input_dir, input_dir))

		return {
			'required': {
				'bind': ('BMAB bind',),
				'control_net_name': (folder_paths.get_filename_list('controlnet'),),
				'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}),
				'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
				'end_percent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
				'image': (files, {'image_upload': True}),
			},
			'optional': {
				'image_in': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind',)
	RETURN_NAMES = ('BMAB bind',)
	FUNCTION = 'apply_controlnet'

	CATEGORY = 'BMAB/controlnet'

	def load_image(self, image):
		image_path = folder_paths.get_annotated_filepath(image)
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

		return output_image, output_mask

	def load_controlnet(self, control_net_name):
		controlnet_path = folder_paths.get_full_path('controlnet', control_net_name)
		controlnet = comfy.controlnet.load_controlnet(controlnet_path)
		return controlnet

	def apply_controlnet(self, bind: BMABBind, control_net_name, strength, start_percent, end_percent, image, **kwargs):
		control_net = self.load_controlnet(control_net_name)

		image_in = kwargs.get('image_in')
		if image_in is None:
			print('NONE image use file.')
			output_image, output_mask = self.load_image(image)
			bgimg = output_image
		else:
			bgimg = image_in

		control_hint = bgimg.movedim(-1, 1)
		cnets = {}

		out = []
		for conditioning in [bind.positive, bind.negative]:
			c = []
			for t in conditioning:
				d = t[1].copy()

				prev_cnet = d.get('control', None)
				if prev_cnet in cnets:
					c_net = cnets[prev_cnet]
				else:
					c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent))
					c_net.set_previous_controlnet(prev_cnet)
					cnets[prev_cnet] = c_net

				d['control'] = c_net
				d['control_apply_to_uncond'] = False
				n = [t[0], d]
				c.append(n)
			out.append(c)
		bind = bind.copy()
		bind.positive = out[0]
		bind.negative = out[1]
		return bind,


class BMABControlNetOpenpose(BMABControlNet):

	def __init__(self) -> None:
		super().__init__()

		self.case = None
		self.cache = None

	def changed(self, c):
		if self.case is None:
			return True
		return not all((a == b for a, b in zip(self.case, c)))

	@classmethod
	def INPUT_TYPES(s):
		input_dir = folder_paths.get_input_directory()
		files = utils.get_file_list(input_dir, input_dir)
		try:
			from comfyui_controlnet_aux.node_wrappers.openpose import OpenPose_Preprocessor
			return {
				'required': {
					'bind': ('BMAB bind',),
					'control_net_name': (s.get_openpose_filenames(),),
					'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}),
					'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
					'end_percent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
					'detect_hand': (["enable", "disable"], {"default": "enable"}),
					'detect_body': (["enable", "disable"], {"default": "enable"}),
					'detect_face': (["enable", "disable"], {"default": "enable"}),
					'fit_to_latent': (["enable", "disable"], {"default": "enable"}),
					'image': (files, {'image_upload': True}),
				}
			}
		except:
			print('failed to load comfyui_controlnet_aux')

		return {
			'required': {
				'text': (
					'STRING',
					{
						'default': 'Cannot Load comfyui_controlnet_aux. To use this node, install comfyui_controlnet_aux',
						'multiline': True,
						'dynamicPrompts': True
					}
				),
			}
		}

	@staticmethod
	def get_openpose_filenames():
		return [cn for cn in folder_paths.get_filename_list('controlnet') if cn.find('openpose') >= 0]

	def apply_controlnet(self, bind: BMABBind, control_net_name, strength, start_percent, end_percent, image, **kwargs):
		from comfyui_controlnet_aux.node_wrappers.openpose import OpenPose_Preprocessor
		detect_hand, detect_body, detect_face = kwargs.get('detect_hand'), kwargs.get('detect_body'), kwargs.get('detect_face')
		fit_to_latent = kwargs.get('fit_to_latent', 'enable') == 'enable'
		c = (image, detect_hand, detect_body, detect_face, fit_to_latent)
		if image is None and 'image_in' in kwargs:
			return super().apply_controlnet(bind, control_net_name, strength, start_percent, end_percent, image, **kwargs)
		if self.changed(c):
			bgimg, _ = self.load_image(image)

			if fit_to_latent:
				w, h = utils.get_shape(bind.latent_image)
				print('Pose image fit to ', w, h)
				results = []
				for img in utils.get_pils_from_pixels(bgimg):
					results.append(utils.resize_and_fill(img, w, h, fill_black=True))
				bgimg = utils.get_pixels_from_pils(results)

			prepro = OpenPose_Preprocessor()
			r = prepro.estimate_pose(bgimg, detect_hand, detect_body, detect_face, kwargs.get('resolution'))
			pixels = r['result'][0]
			self.case = c
			self.cache = pixels
		else:
			pixels = self.cache
		return super().apply_controlnet(bind, control_net_name, strength, start_percent, end_percent, image, image_in=pixels, **kwargs)


class BMABControlNetIPAdapter(BMABControlNet):

	def __init__(self) -> None:
		super().__init__()

		self.case = None
		self.cache = None
		self.annotate_image = None

	def changed(self, c):
		if self.case is None:
			return True
		return not all((a == b for a, b in zip(self.case, c)))

	@classmethod
	def INPUT_TYPES(s):
		input_dir = folder_paths.get_input_directory()
		files = utils.get_file_list(input_dir, input_dir)

		try:
			from ComfyUI_IPAdapter_plus import IPAdapterPlus
			return {
				'required': {
					'bind': ('BMAB bind',),
					'ipadapter_file': (folder_paths.get_filename_list('ipadapter'),),
					'clip_name': (folder_paths.get_filename_list('clip_vision'), ),
					'weight': ('FLOAT', {'default': 1.0, 'min': -1, 'max': 5, 'step': 0.05}),
					'weight_type': (IPAdapterPlus. WEIGHT_TYPES,),
					'combine_embeds': (['concat', 'add', 'subtract', 'average', 'norm average'],),
					'start_at': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
					'end_at': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
					'embeds_scaling': (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
					'resolution': ('INT', {'default': 512, 'min': 128, 'max': 1024, 'step': 8}),
					'fill_noise': (('disable', 'enable'), ),
					'image': (files, {'image_upload': True}),
				},
				'optional': {
					'image_in': ('IMAGE', ),
				}
			}

		except:
			print('failed to load ComfyUI_IPAdapter_plus')

		return {
			'required': {
				'text': (
					'STRING',
					{
						'default': 'Cannot Load ComfyUI_IPAdapter_plus. To use this node, install ComfyUI_IPAdapter_plus',
						'multiline': True,
						'dynamicPrompts': True
					}
				),
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE', )
	RETURN_NAMES = ('BMAB bind', 'annotation', )

	FUNCTION = 'apply_ipadapter'

	def load_ipadapter_model(self, ipadapter_file):
		from ComfyUI_IPAdapter_plus.utils import ipadapter_model_loader
		ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_file)
		return ipadapter_model_loader(ipadapter_file)

	def load_clip(self, clip_name):
		clip_path = folder_paths.get_full_path("clip_vision", clip_name)
		clip_vision = comfy.clip_vision.load(clip_path)
		return clip_vision

	def resize_and_fill(self, bgimg, resolution):
		width, height = resolution, resolution
		resized = Image.new('RGB', (width, height), 0)

		mask = Image.new('L', (width, height), 0)
		dr = ImageDraw.Draw(mask, 'L')

		iratio = width / height
		cratio = bgimg.width / bgimg.height
		if iratio < cratio:
			ratio = width / bgimg.width
			w, h = int(bgimg.width * ratio), int(bgimg.height * ratio)
			y0 = (height - h) // 2
			dr.rectangle((0, y0, w, y0 + h), fill=255)
			resized.paste(bgimg.resize((w, h), Image.Resampling.LANCZOS), (0, y0))
		else:
			ratio = height / bgimg.height
			w, h = int(bgimg.width * ratio), int(bgimg.height * ratio)
			x0 = (width - w) // 2
			dr.rectangle((x0, 0, x0 + w, h), fill=255)
			resized.paste(bgimg.resize((w, h), Image.Resampling.LANCZOS), (x0, 0))

		return resized, mask

	@staticmethod
	def generate_noise(seed, width, height):
		img_1 = np.zeros([height, width, 3], dtype=np.uint8)
		# Generate random Gaussian noise
		mean = 0
		stddev = 180
		r, g, b = cv2.split(img_1)
		# cv2.setRNGSeed(seed)
		cv2.randn(r, mean, stddev)
		cv2.randn(g, mean, stddev)
		cv2.randn(b, mean, stddev)
		img = cv2.merge([r, g, b])
		pil_image = Image.fromarray(img, mode='RGB')
		return pil_image.convert('L').convert('RGB')

	def remove_background(self, image):
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

		blank = self.generate_noise(0, image.width, image.height)
		blank.paste(image.convert('RGBA'), (0, 0), mask=pil_im)
		return blank

	def apply_ipadapter(self, bind: BMABBind, ipadapter_file, clip_name, weight, weight_type, combine_embeds, start_at, end_at, embeds_scaling, resolution, fill_noise, image, image_in=None):
		from ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterAdvanced

		c = (bind.model, ipadapter_file, clip_name, weight, weight_type, combine_embeds, start_at, end_at, embeds_scaling, resolution, fill_noise, image)
		if self.case != c or image_in is not None:
			self.case = c
			fill_noise = fill_noise == 'enable'

			if image_in is not None:
				pixels = image_in
			else:
				pixels, mask = self.load_image(image)
			bgimg = utils.tensor2pil(pixels).convert('RGB')
			resized = utils.resize_and_fill(bgimg, resolution, resolution)
			if fill_noise:
				resized = self.remove_background(resized)
			pixels = utils.pil2tensor(resized)
			self.annotate_image = pixels

			ipadapter = IPAdapterAdvanced()
			ipadapter_model = self.load_ipadapter_model(ipadapter_file)
			clip = self.load_clip(clip_name)
			work_model, face_image = ipadapter.apply_ipadapter(bind.model, ipadapter_model, start_at, end_at, weight, weight_type=weight_type, combine_embeds=combine_embeds, embeds_scaling=embeds_scaling, image=pixels, clip_vision=clip)
			self.cache = work_model
		bind.model = self.cache
		return (bind, self.annotate_image, )



class BMABFluxControlNet:

	@classmethod
	def INPUT_TYPES(s):
		input_dir = folder_paths.get_input_directory()
		files = ['None']
		files.extend(utils.get_file_list(input_dir, input_dir))

		return {
			'required': {
				'bind': ('BMAB bind',),
				"control_net": ("CONTROL_NET",),
				'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}),
				'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
				'end_percent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
				'image': (files, {'image_upload': True}),
			},
			'optional': {
				'image_in': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind',)
	RETURN_NAMES = ('BMAB bind',)
	FUNCTION = 'apply_controlnet'

	CATEGORY = 'BMAB/controlnet'

	def load_image(self, image):
		image_path = folder_paths.get_annotated_filepath(image)
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

		return output_image, output_mask

	def apply_controlnet(self, bind: BMABBind, control_net, strength, start_percent, end_percent, image, **kwargs):

		image_in = kwargs.get('image_in')
		if image_in is None:
			print('NONE image use file.')
			output_image, output_mask = self.load_image(image)
			bgimg = output_image
		else:
			bgimg = image_in

		control_hint = bgimg.movedim(-1, 1)
		cnets = {}

		out = []
		for conditioning in [bind.positive, bind.negative]:
			c = []
			for t in conditioning:
				d = t[1].copy()

				prev_cnet = d.get('control', None)
				if prev_cnet in cnets:
					c_net = cnets[prev_cnet]
				else:
					c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), bind.vae)
					c_net.set_previous_controlnet(prev_cnet)
					cnets[prev_cnet] = c_net

				d['control'] = c_net
				d['control_apply_to_uncond'] = False
				n = [t[0], d]
				c.append(n)
			out.append(c)

		bind = bind.copy()
		bind.positive = out[0]
		bind.negative = out[1]
		return bind,
