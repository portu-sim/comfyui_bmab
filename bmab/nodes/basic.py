import os
import math
import cv2
import json
import numpy as np
import folder_paths
from comfy.cli_args import args
from PIL.PngImagePlugin import PngInfo

from PIL import Image, ImageEnhance, ImageOps
from bmab import utils
from bmab.nodes.binder import BMABBind


def calc_color_temperature(temp):
	white = (255.0, 254.11008387561782, 250.0419083427406)

	temperature = temp / 100

	if temperature <= 66:
		red = 255.0
	else:
		red = float(temperature - 60)
		red = 329.698727446 * math.pow(red, -0.1332047592)
		if red < 0:
			red = 0
		if red > 255:
			red = 255

	if temperature <= 66:
		green = temperature
		green = 99.4708025861 * math.log(green) - 161.1195681661
	else:
		green = float(temperature - 60)
		green = 288.1221695283 * math.pow(green, -0.0755148492)
	if green < 0:
		green = 0
	if green > 255:
		green = 255

	if temperature >= 66:
		blue = 255.0
	else:
		if temperature <= 19:
			blue = 0.0
		else:
			blue = float(temperature - 10)
			blue = 138.5177312231 * math.log(blue) - 305.0447927307
			if blue < 0:
				blue = 0
			if blue > 255:
				blue = 255

	return red / white[0], green / white[1], blue / white[2]


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


class BMABBasic:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'contrast': ('FLOAT', {'default': 1.0, 'min': 0, 'max': 2, 'step': 0.05}),
				'brightness': ('FLOAT', {'default': 1.0, 'min': 0, 'max': 2, 'step': 0.05}),
				'sharpeness': ('FLOAT', {'default': 1.0, 'min': -5.0, 'max': 5.0, 'step': 0.1}),
				'color_saturation': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 2.0, 'step': 0.01}),
				'color_temperature': ('INT', {'default': 0, 'min': -2000, 'max': 2000, 'step': 1}),
				'noise_alpha': ('FLOAT', {'default': 0, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
			},
			'optional': {
				'bind': ('BMAB bind',),
				'image': ('IMAGE',),
			},
			'hidden': {'unique_id': 'UNIQUE_ID'}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE', )
	RETURN_NAMES = ('BMAB bind', 'image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/basic'

	def process(self, contrast, brightness, sharpeness, color_saturation, color_temperature, noise_alpha, unique_id, bind: BMABBind=None, image=None):
		if bind is None:
			pixels = image
		else:
			pixels = bind.pixels if image is None else image

		results = []
		for bgimg in utils.get_pils_from_pixels(pixels):
			if contrast != 1:
				enhancer = ImageEnhance.Contrast(bgimg)
				bgimg = enhancer.enhance(contrast)

			if brightness != 1:
				enhancer = ImageEnhance.Brightness(bgimg)
				bgimg = enhancer.enhance(brightness)

			if sharpeness != 1:
				enhancer = ImageEnhance.Sharpness(bgimg)
				bgimg = enhancer.enhance(sharpeness)

			if color_saturation != 1:
				enhancer = ImageEnhance.Color(bgimg)
				bgimg = enhancer.enhance(color_saturation)

			if color_temperature != 0:
				temp = calc_color_temperature(6500 + color_temperature)
				az = []
				for d in bgimg.getdata():
					az.append((int(d[0] * temp[0]), int(d[1] * temp[1]), int(d[2] * temp[2])))
				bgimg = Image.new('RGB', bgimg.size)
				bgimg.putdata(az)

			if noise_alpha != 0:
				img_noise = utils.generate_noise(0, bgimg.width, bgimg.height)
				bgimg = Image.blend(bgimg, img_noise, alpha=noise_alpha)

			results.append(bgimg)

		pixels = utils.get_pixels_from_pils(results)
		return BMABBind.result(bind, pixels, )


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

	def process(self, pixels, threshold1, threshold2, strength, unique_id):
		results = []
		for bgimg in utils.get_pils_from_pixels(pixels):
			bgimg = edge_flavor(bgimg, threshold1, threshold2, strength)
			results.append(bgimg)
		pixels = utils.pil2tensor(results)
		return (pixels,)


class BMABSaveImage:
	def __init__(self):
		self.output_dir = folder_paths.get_output_directory()
		self.type = 'output'
		self.prefix_append = ''
		self.compress_level = 4

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'filename_prefix': ('STRING', {'default': 'ComfyUI'})
			},
			'hidden': {
				'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'
			},
			'optional': {
				'bind': ('BMAB bind',),
				'images': ('IMAGE',),
			}
		}

	RETURN_TYPES = ()
	FUNCTION = 'save_images'

	OUTPUT_NODE = True

	CATEGORY = 'BMAB/basic'

	def save_images(self, filename_prefix='ComfyUI', prompt=None, extra_pnginfo=None, bind: BMABBind=None, images=None):
		if images is None:
			images = bind.pixels
		filename_prefix += self.prefix_append
		full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
		results = list()
		for (batch_number, image) in enumerate(images):
			i = 255. * image.cpu().numpy()
			img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
			metadata = None
			if not args.disable_metadata:
				metadata = PngInfo()
				if prompt is not None:
					metadata.add_text('prompt', json.dumps(prompt))
				if extra_pnginfo is not None:
					for x in extra_pnginfo:
						metadata.add_text(x, json.dumps(extra_pnginfo[x]))

			filename_with_batch_num = filename.replace('%batch_num%', str(batch_number))
			if bind is not None:
				file = f'{filename_with_batch_num}_{counter:05}_{bind.seed}_.png'
			else:
				file = f'{filename_with_batch_num}_{counter:05}_.png'
			img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
			results.append({
				'filename': file,
				'subfolder': subfolder,
				'type': self.type
			})
			counter += 1

		return {'ui': {'images': results}}
