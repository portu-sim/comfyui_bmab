import torch

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

import comfy
import nodes
import folder_paths

from bmab import utils
from bmab.nodes import BMABBind
from bmab.utils.color import apply_color_correction


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


def load_controlnet(control_net_name):
	controlnet_path = folder_paths.get_full_path('controlnet', control_net_name)
	controlnet = comfy.controlnet.load_controlnet(controlnet_path)
	return controlnet


def apply_controlnet(control_net_name, positive, negative, strength, start_percent, end_percent, image):
	control_net = load_controlnet(control_net_name)

	control_hint = image.movedim(-1, 1)
	cnets = {}

	out = []
	for conditioning in [positive, negative]:
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
	return out[0], out[1]


def preprocess(image, mask):
	mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(image.shape[1], image.shape[2]), mode="bilinear")
	mask = mask.movedim(1, -1).expand((-1, -1, -1, 3))
	image = image.clone()
	image[mask > 0.5] = -1.0
	return image


def process_img2img_with_controlnet(bind: BMABBind, image, params, controlnet_name, mask=None):
	steps, cfg, sampler_name, scheduler, denoise = params['steps'], params['cfg_scale'], params['sampler_name'], params['scheduler'], params['denoise']

	pixels = utils.pil2tensor(image.convert('RGB'))
	latent = dict(samples=bind.vae.encode(pixels))

	if mask is not None:
		mask_pixels = utils.pil2tensor(mask.convert('RGB'))
		cn_pixels = preprocess(pixels, mask_pixels[:, :, :, 0])
	else:
		cn_pixels = pixels
	positive, negative = apply_controlnet(controlnet_name, bind.positive, bind.negative, 1.0, 0.0, 1.0, cn_pixels)

	samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)[0]
	latent = bind.vae.decode(samples["samples"])
	result = utils.tensor2pil(latent)
	return result


def process_img2img_with_controlnet_mask(bind: BMABBind, controlnet_name, image, params, mask=None, box=None):
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
	processed = process_img2img_with_controlnet(bind, resized, params, controlnet_name, mask=mask)
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






