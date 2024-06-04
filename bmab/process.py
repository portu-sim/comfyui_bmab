import cv2
import numpy as np
from skimage import exposure
from blendmodes.blend import blendLayers, BlendType

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

import nodes
from bmab import utils
from bmab.nodes import BMABBind


def apply_color_correction(correction, original_image):
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


def process_img2img(bind: BMABBind, image, params):
	steps, cfg, sampler_name, scheduler, denoise = params['steps'], params['cfg_scale'], params['sampler_name'], params['scheduler'], params['denoise']
	pixels = utils.pil2tensor(image.convert('RGB'))
	latent = dict(samples=bind.vae.encode(pixels))
	samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise)[0]
	latent = bind.vae.decode(samples["samples"])
	result = utils.tensor2pil(latent)
	return result


def process_img2img_with_mask(bind: BMABBind, image, params, mask=None, box=None):
	width, height, padding, dilation = params['width'], params['height'], params['padding'], params['dilation']

	if box is None:
		box = mask.getbbox()
		if box is None:
			return image

	if mask is None:
		mask = Image.new('L', image.size, 0)
		dr = ImageDraw.Draw(mask, 'L')
		dr.rectangle(box, fill=255)

	x1, y1, x2, y2 = tuple(int(x) for x in box)

	cbx = utils.get_box_with_padding(image, (x1, y1, x2, y2), padding)
	cropped = image.crop(cbx)
	resized = utils.resize_and_fill(cropped, width, height)
	processed = process_img2img(bind, resized, params)
	processed = apply_color_correction(resized, processed)

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

	pil_mask = utils.dilate_mask(mask, dilation)
	blur = ImageFilter.GaussianBlur(dilation)
	blur_mask = pil_mask.filter(blur)

	image.paste(img, (0, 0), mask=blur_mask)
	return image
