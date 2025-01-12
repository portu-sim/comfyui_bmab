import os
import math
import json
import glob
import time
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

	RETURN_TYPES = ('BMAB bind', 'IMAGE',)
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/basic'

	def process(self, contrast, brightness, sharpeness, color_saturation, color_temperature, noise_alpha, unique_id, bind: BMABBind = None, image=None):
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


class BMABSaveImage:
	def __init__(self):
		self.output_dir = folder_paths.get_output_directory()
		self.type = 'output'
		self.compress_level = 4

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'filename_prefix': ('STRING', {'default': 'bmab'}),
				'format': (['png', 'jpg'], ),
				'use_date': (['disable', 'enable'], ),
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

	@staticmethod
	def get_file_sequence(prefix, subdir):
		output_dir = os.path.normpath(os.path.join(folder_paths.get_output_directory(), subdir))
		find_path = os.path.join(output_dir, f'{prefix}*')
		sequence = 0
		for f in glob.glob(find_path):
			filename = os.path.basename(f)
			split_name = filename[len(prefix)+1:].replace('.', '_').split('_')
			try:
				file_sequence = int(split_name[0])
			except:
				continue
			if file_sequence > sequence:
				sequence = file_sequence
		return sequence + 1

	@staticmethod
	def get_sub_directory(use_date):
		if not use_date:
			return ''

		dd = time.strftime('%Y-%m-%d', time.localtime(time.time()))
		full_output_folder = os.path.join(folder_paths.output_directory, dd)
		print(full_output_folder)
		if not os.path.exists(full_output_folder):
			os.mkdir(full_output_folder)
		return dd

	def save_images(self, filename_prefix='bmab', format='png', use_date='disable', prompt=None, extra_pnginfo=None, bind: BMABBind = None, images=None):
		if images is None:
			images = bind.pixels
		output_dir = folder_paths.get_output_directory()
		results = list()
		use_date = use_date == 'enable'

		subdir = self.get_sub_directory(use_date)
		prefix_split = filename_prefix.split('/')
		if len(prefix_split) != 1:
			filename_prefix = prefix_split[-1]
			subdir = os.path.join(subdir, '/'.join(prefix_split[:-1]))
		sequence = self.get_file_sequence(filename_prefix, subdir)

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

			if batch_number > 0:
				filename = f'{filename_prefix}_{sequence:05}_{batch_number}'
			else:
				filename = f'{filename_prefix}_{sequence:05}'

			if bind is not None:
				file = f'{filename}_{bind.seed}.{format}'
			else:
				file = f'{filename}.{format}'

			if use_date:
				output_dir = os.path.join(output_dir, subdir)

			if not os.path.exists(output_dir):
				os.mkdir(output_dir)

			if format == 'png':
				img.save(os.path.join(output_dir, file), pnginfo=metadata, compress_level=self.compress_level)
			else:
				img.save(os.path.join(output_dir, file))

			results.append({
				'filename': file,
				'subfolder': subdir,
				'type': self.type
			})

			sequence += 1

		return {'ui': {'images': results}}


class BMABText:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
			},
			'optional': {
				'text': ('STRING', {"forceInput": True}),
			}
		}

	RETURN_TYPES = ('STRING',)
	RETURN_NAMES = ('string',)
	FUNCTION = 'export'

	CATEGORY = 'BMAB/basic'

	def export(self, prompt, text=None):
		if text is not None:
			prompt = prompt.replace('__prompt__', text)
		result = utils.parse_prompt(prompt, 0)
		return (result,)


class BMABPreviewText:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"text": ("STRING", {"forceInput": True}),
			},
			"hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
		}

	RETURN_TYPES = ("STRING",)
	OUTPUT_NODE = True
	FUNCTION = "preview_text"

	CATEGORY = "BMAB/basic"

	def preview_text(self, text, prompt=None, extra_pnginfo=None):
		return {"ui": {"string": [text, ]}, "result": (text,)}


class BMABRemoteAccessAndSave(BMABSaveImage):

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'filename_prefix': ('STRING', {'default': 'bmab'}),
				'format': (['png', 'jpg'], ),
				'use_date': (['disable', 'enable'], ),
				'remote_name': ('STRING', {'multiline': False}),
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
	FUNCTION = 'remote_save_images'

	OUTPUT_NODE = True

	CATEGORY = 'BMAB/basic'

	def remote_save_images(self, filename_prefix='bmab', format='png', use_date='disable', remote_name=None, prompt=None, extra_pnginfo=None, bind: BMABBind = None, images=None):
		return self.save_images(filename_prefix, format, use_date, prompt, extra_pnginfo, bind, images)


