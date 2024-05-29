import os
import cv2
import numpy as np

from PIL import Image

import bmab
from bmab import utils
from bmab.nodes.binder import BMABBind


class BMABModelToBind:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
			},
			'optional': {
				'model': ('MODEL',),
				'clip': ('CLIP',),
				'vae': ('VAE',),
			}
		}

	RETURN_TYPES = ('BMAB bind', )
	RETURN_NAMES = ('bind', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/utils'

	def process(self, bind: BMABBind, model=None, clip=None, vae=None):
		if model is not None:
			bind.model = model
		if clip is not None:
			bind.clip = clip
		if vae is not None:
			bind.vae = vae
		return (bind, )


class BMABConditioningToBind:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
			},
			'optional': {
				"positive": ("CONDITIONING",),
				"negative": ("CONDITIONING",),
			}
		}

	RETURN_TYPES = ('BMAB bind', )
	RETURN_NAMES = ('bind', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/utils'

	def process(self, bind: BMABBind, positive=None, negative=None):
		if positive is not None:
			bind.positive = positive
		if negative is not None:
			bind.negative = negative
		return (bind, )


class BMABNoiseGenerator:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'width': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 8}),
			},
			'optional': {
				'bind': ('BMAB bind',),
				'latent': ('LATENT',),
			}
		}

	RETURN_TYPES = ('IMAGE', )
	RETURN_NAMES = ('image', )
	FUNCTION = 'generate'

	CATEGORY = 'BMAB/utils'

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
		return pil_image

	def generate(self, width, height, bind: BMABBind=None, latent=None):
		if bind is not None:
			width, height = utils.get_shape(bind.latent_image)
		if latent is not None:
			width, height = utils.get_shape(latent)

		cache_path = os.path.join(os.path.dirname(bmab.__file__), '../resources/cache')
		filename = f'noise_{width}_{height}.png'
		full_path = os.path.join(cache_path, filename)
		if os.path.exists(full_path) and os.path.isfile(full_path):
			noise = Image.open(full_path)
			return (utils.get_pixels_from_pils([noise]), )

		noise = self.generate_noise(0, width, height)
		noise.save(full_path)
		return (utils.get_pixels_from_pils([noise]),)

