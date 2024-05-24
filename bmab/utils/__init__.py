import os
import gc
import sys
import cv2
import torch
import numpy as np
import glob
import importlib.util

from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter


def pil2tensor(image):
	i = np.array(image).astype(np.float32) / 255.0
	return torch.from_numpy(i).unsqueeze(0)


def merge(t1, t2):
	torch.concat([t1, t2])


def tensor2pil(image):
	return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
	if model_dir is None:
		hub_dir = get_dir()
		model_dir = os.path.join(hub_dir, 'checkpoints')

	os.makedirs(model_dir, exist_ok=True)

	parts = urlparse(url)
	filename = os.path.basename(parts.path)
	if file_name is not None:
		filename = file_name
	cached_file = os.path.abspath(os.path.join(model_dir, filename))
	if not os.path.exists(cached_file):
		print(f'Downloading: "{url}" to {cached_file}\n')
		download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
	return cached_file


def lazy_loader(filename):
	bmab_model_path = os.path.join(os.path.dirname(__file__), '../../models')
	files = glob.glob(bmab_model_path)

	targets = {
		'sam_vit_b_01ec64.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
		'sam_vit_l_0b3195.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
		'sam_vit_h_4b8939.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
		'groundingdino_swint_ogc.pth': 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth',
		'GroundingDINO_SwinT_OGC.py': 'https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py',
		'face_yolov8n.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt',
		'face_yolov8n_v2.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt',
		'face_yolov8m.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt',
		'face_yolov8s.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt',
		'hand_yolov8n.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt',
		'hand_yolov8s.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt',
		'person_yolov8m-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt',
		'person_yolov8n-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt',
		'person_yolov8s-seg.pt': 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt',
		'sam_hq_vit_b.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth',
		'sam_hq_vit_h.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth',
		'sam_hq_vit_l.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth',
		'sam_hq_vit_tiny.pth': 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth',
		'bmab_face_nm_yolov8n.pt': 'https://huggingface.co/portu-sim/bmab/resolve/main/bmab_face_nm_yolov8n.pt',
		'bmab_face_sm_yolov8n.pt': 'https://huggingface.co/portu-sim/bmab/resolve/main/bmab_face_sm_yolov8n.pt',
		'bmab_hand_yolov8n.pt': 'https://huggingface.co/portu-sim/bmab/resolve/main/bmab_hand_yolov8n.pt',
		'ControlNetLama.pth': 'https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth',
	}

	file = os.path.join(bmab_model_path, filename)
	if os.path.exists(file):
		return file

	if filename in targets and filename not in files:
		load_file_from_url(targets[filename], bmab_model_path)
	return file


def list_pretraining_models():
	bmab_model_path = os.path.join(os.path.dirname(__file__), '../../models')
	files = glob.glob(os.path.join(bmab_model_path, '*.pt'))
	return [os.path.basename(f) for f in files]


def load_pretraining_model(filename):
	bmab_model_path = os.path.join(os.path.dirname(__file__), '../../models')
	return os.path.join(bmab_model_path, filename)


def torch_gc():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()


def debug_save_image(img, name):
	bmab_model_path = os.path.join(os.path.dirname(__file__), f'../../{name}')
	img.save(bmab_model_path)


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


alignment = {
	'top': lambda dx, dy: (dx / 2, dx / 2, 0, dy),
	'top-right': lambda dx, dy: (dx, 0, 0, dy),
	'right': lambda dx, dy: (dx, 0, dy / 2, dy / 2),
	'bottom-right': lambda dx, dy: (dx, 0, dy, 0),
	'bottom': lambda dx, dy: (dx / 2, dx / 2, dy, 0),
	'bottom-left': lambda dx, dy: (0, dx, dy, 0),
	'left': lambda dx, dy: (0, dx, dy / 2, dy / 2),
	'top-left': lambda dx, dy: (0, dx, 0, dy),
	'center': lambda dx, dy: (dx / 2, dx / 2, dy / 2, dy / 2),
}


def resize_image_with_alignment(image, al, width, height):
	if al not in alignment:
		return image
	return resize_margin(image, *alignment[al](width - image.width, height - image.height))


def get_mask_with_alignment(image, al, width, height, dilation=0):
	return draw_mask(image, *alignment[al](width - image.width, height - image.height), dilation)


def resize_margin(image, left, right, top, bottom):
	left = int(left)
	right = int(right)
	top = int(top)
	bottom = int(bottom)
	input_image = image.copy()

	if left != 0:
		res = Image.new("RGB", (image.width + left, image.height))
		res.paste(image, (left, 0))
		res.paste(image.resize((left, image.height), box=(0, 0, 0, image.height)), box=(0, 0))
		image = res
	if right != 0:
		res = Image.new("RGB", (image.width + right, image.height))
		res.paste(image, (0, 0))
		res.paste(image.resize((right, image.height), box=(image.width, 0, image.width, image.height)), box=(image.width, 0))
		image = res
	if top != 0:
		res = Image.new("RGB", (image.width, image.height + top))
		res.paste(image, (0, top))
		res.paste(image.resize((image.width, top), box=(0, 0, image.width, 0)), box=(0, 0))
		image = res
	if bottom != 0:
		res = Image.new("RGB", (image.width, image.height + bottom))
		res.paste(image, (0, 0))
		res.paste(image.resize((image.width, bottom), box=(0, image.height, image.width, image.height)), box=(0, image.height))
		image = res

	img = image.filter(ImageFilter.GaussianBlur(10))
	region_size = 10
	width, height = img.size
	for y in range(0, height, region_size):
		for x in range(0, width, region_size):
			region = img.crop((x, y, x + region_size, y + region_size))
			average_color = region.resize((1, 1)).getpixel((0, 0))
			img.paste(average_color, (x, y, x + region_size, y + region_size))
	img.paste(input_image, box=(left, top))
	image = img.resize(input_image.size, resample=Image.Resampling.LANCZOS)
	return image


def draw_mask(image, left, right, top, bottom, d=0):
	left = int(left)
	right = int(right)
	top = int(top)
	bottom = int(bottom)

	width = image.width + left + right
	height = image.height + top + bottom

	box = (left + d, top + d, left + image.width - d, top + image.height - d)
	mask = Image.new('L', (width, height), 255)
	dr = ImageDraw.Draw(mask, 'L')
	dr.rectangle(box, fill=0)
	mask = mask.resize(image.size, resample=Image.Resampling.LANCZOS)
	return mask, box


def fix_size_by_scale(w, h, scale):
	w = int(((w * scale) // 8) * 8)
	h = int(((h * scale) // 8) * 8)
	return w, h


def fix_box_by_scale(box, scale):
	x1, y1, x2, y2 = tuple(int(x) for x in box)
	w = x2 - x1
	h = y2 - y1
	dx = int(w * scale / 2)
	dy = int(h * scale / 2)
	return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def fix_box_size(box):
	x1, y1, x2, y2 = tuple(int(x) for x in box)
	w = x2 - x1
	h = y2 - y1
	w = (w // 8) * 8
	h = (h // 8) * 8
	return x1, y1, x1 + w, y1 + h


def fix_box_limit(box, size):
	x1, y1, x2, y2 = tuple(int(x) for x in box)
	w = size[0]
	h = size[1]
	if x1 < 0:
		x1 = 0
	if y1 < 0:
		y1 = 0
	if x2 >= w:
		x2 = w - 1
	if y2 >= h:
		y2 = h - 1
	return x1, y1, x2, y2


def get_box_with_padding(mask, box, pad=0):
	x1, y1, x2, y2 = box
	return (max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, mask.size[0]), min(y2 + pad, mask.size[1])) if pad else box


def resize_and_fill(im, width, height, fill_black=False):
	ratio = width / height
	src_ratio = im.width / im.height

	src_w = width if ratio < src_ratio else im.width * height // im.height
	src_h = height if ratio >= src_ratio else im.height * width // im.width

	resized = im.resize((src_w, src_h), resample=Image.Resampling.LANCZOS)
	res = Image.new("RGB", (width, height))
	res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

	if not fill_black:
		if ratio < src_ratio:
			fill_height = height // 2 - src_h // 2
			if fill_height > 0:
				res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
				res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
		elif ratio > src_ratio:
			fill_width = width // 2 - src_w // 2
			if fill_width > 0:
				res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
				res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))
	return res


def crop(image, width, height, resized=True):
	# if image.width != width and image.height != height:
	#	raise ValueError('Image not matched')
	iratio = image.width / image.height
	cratio = width / height
	if iratio < cratio:
		if resized:
			ratio = width / image.width
			image = image.resize((int(image.width * ratio), int(image.height * ratio)))
		else:
			width, height = image.width, int(image.width * height / width)
		y0 = (image.height - height) // 2
		image = image.crop((0, y0, width, y0 + height))
	else:
		if resized:
			ratio = height / image.height
			image = image.resize((int(image.width * ratio), int(image.height * ratio)))
		else:
			width, height = int(image.height * width / height), image.height
		x0 = (image.width - width) // 2
		image = image.crop((x0, 0, x0 + width, height))
	return image


resource_path = os.path.join(os.path.dirname(__file__), '../../resources')
model_path = os.path.join(os.path.dirname(__file__), '../../models')


def parse_prompt(prompt: str, seed):
	idx = 0
	for i in range(0, 100):
		start = prompt.find('__', idx)
		if start >= 0:
			idx = start + 2
			end = prompt.find('__', idx)
			if end >= 0:
				idx = start
				file = os.path.join(resource_path, f'wildcard/{prompt[start + 2:end].strip()}.txt')
				if os.path.exists(file):
					with open(file) as f:
						lines = f.readlines()
					candidate = [x for x in [l.strip() for l in lines] if len(x) > 0]
					length = len(candidate)
					prompt = prompt[:start] + candidate[seed % length] + prompt[end + 2:]
				else:
					print(f'Not found wildcard {prompt[start + 2:end].strip()}.txt')
					prompt = prompt[:start] + prompt[end + 2:]
			else:
				break
		else:
			break
	return prompt


def get_cache_path(filename):
	return os.path.join(resource_path, f'cache/{filename}')


def load_external_module(module_path, module_name):
	path = os.path.dirname(__file__)
	custom_nodes_path = os.path.join(path, '../../../')
	target = os.path.join(custom_nodes_path, module_path)
	target_path = os.path.normpath(target)
	print('target_path', target_path)
	if not os.path.exists(target_path):
		return None
	return load_module(target_path, module_name)


def load_module(file_name, module_name):
	spec = importlib.util.spec_from_file_location(module_name, file_name)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def get_blur_mask(size, box, dilation):
	mask = Image.new('L', size, 0)
	dr = ImageDraw.Draw(mask, 'L')
	dr.rectangle(box, fill=255)
	if dilation == 0:
		return mask
	blur = ImageFilter.GaussianBlur(dilation)
	return mask.filter(blur)


def get_file_list(base, dd):
	files = []
	for f in glob.glob(f'{dd}/*'):
		if os.path.isdir(f):
			files.extend(get_file_list(base, f))
		else:
			files.append(os.path.relpath(f, base).replace('\\', '/'))
	return files


def get_device():
	if sys.platform == 'darwin':
		return 'mps'
	elif torch.cuda.is_available():
		return 'cuda'
	return 'cpu'


def get_pils_from_pixels(pixels):
	pil_images = []
	for (batch_number, pixels) in enumerate(pixels):
		i = 255. * pixels.cpu().numpy().squeeze()
		img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
		pil_images.append(img)
	return pil_images


def get_pixels_from_pils(pil_imgs):
	pixels = []
	for (batch_number, pil_img) in enumerate(pil_imgs):
		i = np.array(pil_img).astype(np.float32) / 255.0
		pixels.append(i)
	return torch.from_numpy(np.array(pixels))


def dilate_mask(mask, dilation):
	if dilation < 4:
		return mask
	arr = np.array(mask)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
	arr = cv2.dilate(arr, kernel, iterations=1)
	return Image.fromarray(arr)


def is_box_in_box(box1, box2):
	x1, y1, x2, y2 = box1
	cx, cy = x1 + ((x2 - x1) // 2), y1 + ((y2 - y1) // 2)
	x1, y1, x2, y2 = box2
	result = (x1 <= cx <= x2 and y1 <= cy <= y2)
	print('box in box result', result, box1, box2)
	return result


def revert_image(width, height, image, cropped):
	iratio = width / height
	cratio = cropped.width / cropped.height
	if iratio < cratio:
		ratio = cropped.width / width
		image = image.resize((int(image.width * ratio), int(image.height * ratio)))
		y0 = (image.height - cropped.height) // 2
		image = image.crop((0, y0, cropped.width, y0 + cropped.height))
	else:
		ratio = cropped.height / height
		image = image.resize((int(image.width * ratio), int(image.height * ratio)))
		x0 = (image.width - cropped.width) // 2
		image = image.crop((x0, 0, x0 + cropped.width, cropped.height))
	return image


def box_intersection(rect1, rect2):
	x1, y1 = rect1[0], rect1[1]
	x2, y2 = rect1[2], rect1[3]
	x3, y3 = rect2[0], rect2[1]
	x4, y4 = rect2[2], rect2[3]

	if x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4:
		return 0

	left_up_x = max(x1, x3)
	left_up_y = max(y1, y3)
	right_down_x = min(x2, x4)
	right_down_y = min(y2, y4)

	width = right_down_x - left_up_x
	height = right_down_y - left_up_y

	return width * height


def is_samebox(box1, box2):
	return box1[0] == box2[0] and box1[1] == box2[1] and box1[2] == box2[2] and box1[3] == box2[3]
