import os
import sys
import comfy
import torch
import numpy as np
import folder_paths
import node_helpers
from PIL import Image
from PIL import ImageSequence
from PIL import ImageOps
from bmab import utils
from bmab.nodes.binder import BMABBind


class BMABControlNet:

	@classmethod
	def INPUT_TYPES(s):
		input_dir = folder_paths.get_input_directory()
		files = ['None']
		files.extend(sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]))

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
			output_image, output_mask = self.load_image(image_in)
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

	@classmethod
	def INPUT_TYPES(s):
		cnnames = [cn for cn in folder_paths.get_filename_list('controlnet') if cn.find('openpose') >= 0]
		input_dir = folder_paths.get_input_directory()
		files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

		try:
			from comfyui_controlnet_aux.node_wrappers.openpose import OpenPose_Preprocessor
			return {
				'required': {
					'bind': ('BMAB bind',),
					'control_net_name': (cnnames,),
					'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}),
					'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
					'end_percent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
					'detect_hand': (["enable", "disable"], {"default": "enable"}),
					'detect_body': (["enable", "disable"], {"default": "enable"}),
					'detect_face': (["enable", "disable"], {"default": "enable"}),
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

	def apply_controlnet(self, bind: BMABBind, control_net_name, strength, start_percent, end_percent, image, **kwargs):
		from comfyui_controlnet_aux.node_wrappers.openpose import OpenPose_Preprocessor
		bgimg, _ = self.load_image(image)
		prepro = OpenPose_Preprocessor()
		r = prepro.estimate_pose(bgimg, kwargs.get('detect_hand'), kwargs.get('detect_body'), kwargs.get('detect_face'), kwargs.get('resolution'))
		pixels = r['result'][0]
		return super().apply_controlnet(bind, control_net_name, strength, start_percent, end_percent, image, image_in=pixels, **kwargs)

