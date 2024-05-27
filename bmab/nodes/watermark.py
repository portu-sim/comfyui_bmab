import os
import sys
import glob

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from bmab import utils


class BMABWatermark:
	alignment = {
		'bottom-left': lambda w, h, cx, cy: (0, h - cy),
		'top': lambda w, h, cx, cy: (w / 2 - cx / 2, 0),
		'top-right': lambda w, h, cx, cy: (w - cx, 0),
		'right': lambda w, h, cx, cy: (w - cx, h / 2 - cy / 2),
		'bottom-right': lambda w, h, cx, cy: (w - cx, h - cy),
		'bottom': lambda w, h, cx, cy: (w / 2 - cx / 2, h - cy),
		'left': lambda w, h, cx, cy: (0, h / 2 - cy / 2),
		'top-left': lambda w, h, cx, cy: (0, 0),
		'center': lambda w, h, cx, cy: (w / 2 - cx / 2, h / 2 - cy / 2),
	}


	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'font': (s.list_fonts(),),
				'alignment': ([x for x in s.alignment.keys()], ),
				'text_alignment': (['left', 'right', 'center'], ),
				'rotate': ([0, 90, 180, 270], ),
				'color': ('STRING', {'default': '#000000'}),
				'background_color': ('STRING', {'default': '#000000'}),
				'font_size': ('INT', {'default': 12, 'min': 4, 'max': 128}),
				'transparency': ('INT', {'default': 100, 'min': 0, 'max': 100}),
				'background_transparency': ('INT', {'default': 0, 'min': 0, 'max': 100}),
				'margin': ('INT', {'default': 5, 'min': 0, 'max': 100}),
				'text': ('STRING', {'multiline': True}),
			},
			'optional': {
				'bind': ('BMAB bind',),
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE',)
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/basic'

	def process_watermark(self, img, font, alignment, text_alignment, rotate, color, background_color, font_size, transparency, background_transparency, margin, text):
		background_color = self.color_hex_to_rgb(background_color, int(255 * (background_transparency / 100)))

		if os.path.isfile(text):
			cropped = Image.open(text)
		else:
			font = self.get_font(font, font_size)
			color = self.color_hex_to_rgb(color, int(255 * (transparency / 100)))

			# 1st
			base = Image.new('RGBA', img.size, background_color)
			draw = ImageDraw.Draw(base)
			bbox = draw.textbbox((0, 0), text, font=font)
			draw.text((0, 0), text, font=font, fill=color, align=text_alignment)
			cropped = base.crop(bbox)

		# 2st margin
		base = Image.new('RGBA', (cropped.width + margin * 2, cropped.height + margin * 2), background_color)
		base.paste(cropped, (margin, margin))

		# 3rd rotate
		base = base.rotate(rotate, expand=True)

		# 4th
		image = img.convert('RGBA')
		image2 = image.copy()
		x, y = BMABWatermark.alignment[alignment](image.width, image.height, base.width, base.height)
		image2.paste(base, (int(x), int(y)))
		return Image.alpha_composite(image, image2)

	def process(self, image=None, bind=None, **kwargs):
		pixels = bind.pixels if image is None else image
		results = []
		for img in utils.get_pils_from_pixels(pixels):
			results.append(self.process_watermark(img, **kwargs))
		pixels = utils.get_pixels_from_pils(results)
		return (bind, pixels, )

	@staticmethod
	def color_hex_to_rgb(value, transparency):
		value = value.lstrip('#')
		lv = len(value)
		r, g, b = tuple(int(value[i:i + 2], 16) for i in range(0, lv, 2))
		return r, g, b, transparency

	@staticmethod
	def list_fonts():
		if sys.platform == 'win32':
			path = 'C:\\Windows\\Fonts\\*.ttf'
			files = glob.glob(path)
			return [os.path.basename(f) for f in files]
		if sys.platform == 'darwin':
			path = '/System/Library/Fonts/*'
			files = glob.glob(path)
			return [os.path.basename(f) for f in files]
		if sys.platform == 'linux':
			path = '/usr/share/fonts/*'
			files = glob.glob(path)
			fonts = [os.path.basename(f) for f in files]
			if 'SAGEMAKER_INTERNAL_IMAGE_URI' in os.environ:
				path = '/opt/conda/envs/sagemaker-distribution/fonts/*'
				files = glob.glob(path)
				fonts.extend([os.path.basename(f) for f in files])
			return fonts
		return ['']

	@staticmethod
	def get_font(font, size):
		if sys.platform == 'win32':
			path = f'C:\\Windows\\Fonts\\{font}'
			return ImageFont.truetype(path, size, encoding="unic")
		if sys.platform == 'darwin':
			path = f'/System/Library/Fonts/{font}'
			return ImageFont.truetype(path, size, encoding="unic")
		if sys.platform == 'linux':
			if 'SAGEMAKER_INTERNAL_IMAGE_URI' in os.environ:
				path = f'/opt/conda/envs/sagemaker-distribution/fonts/{font}'
			else:
				path = f'/usr/share/fonts/{font}'
			return ImageFont.truetype(path, size, encoding="unic")
