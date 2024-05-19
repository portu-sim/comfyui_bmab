import os
import json
import requests
import copy

import threading
import time
import comfy

from PIL import Image
from bmab import utils

import base64
from io import BytesIO
import folder_paths


def b64_encoding(image):
	buffered = BytesIO()
	image.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode("utf-8")


def b64_decoding(b64):
	return Image.open(BytesIO(base64.b64decode(b64)))


class ApiServer:

	def __init__(self, ipaddr, port) -> None:
		super().__init__()
		self.ipaddr = ipaddr
		self.port = port
		self.sdwebui_info = {}
		self.controlnet_info = {}

	def post(self, uri, data):
		url = f'http://{self.ipaddr}:{self.port}/{uri}'
		return requests.post(url, data=json.dumps(data))

	def get(self, uri):
		url = f'http://{self.ipaddr}:{self.port}/{uri}'
		resp = requests.get(url)
		resp.raise_for_status()
		return resp.json()

	def get_all_info(self):
		try:
			self.update_all_info()
		except:
			print(f'An error occured in API server {self.ipaddr}:{self.port}')

	def update_all_info(self):
		resources = [
			'samplers',
			'schedulers',
			'upscalers',
			'sd-models',
			'sd-vae',
			'loras',
			'options',
		]
		for resource in resources:
			self.sdwebui_info[resource] = self.get(f'sdapi/v1/{resource}')

		self.sdwebui_info['sampler_list'] = [x['name'] for x in self.sdwebui_info['samplers']]
		self.sdwebui_info['scheduler_list'] = [x['label'] for x in self.sdwebui_info['schedulers']]
		self.sdwebui_info['upscaler_list'] = [x['name'] for x in self.sdwebui_info['upscalers']]
		self.sdwebui_info['sd-model_list'] = [x['title'] for x in self.sdwebui_info['sd-models']]
		self.sdwebui_info['sd-vae_list'] = [x['model_name'] for x in self.sdwebui_info['sd-vae']]
		self.sdwebui_info['lora_list'] = [x['name'] for x in self.sdwebui_info['loras']]

		self.controlnet_info['models'] = self.get(f'controlnet/model_list').get('model_list', [])
		self.controlnet_info['modules'] = self.get(f'controlnet/module_list').get('module_list', [])

		with open(utils.get_cache_path('webuiapi.json'), 'w') as f:
			json.dump(self.sdwebui_info, f, indent=2)

		with open(utils.get_cache_path('controlnetapi.json'), 'w') as f:
			json.dump(self.controlnet_info, f, indent=2)

	@classmethod
	def get_upscaler(cls):
		try:
			with open(utils.get_cache_path('webuiapi.json'), 'r') as f:
				j = json.load(f)
				return j.get('upscaler_list', [])
		except:
			print(f'An error occured in API server.')
			return []

	@classmethod
	def get_sampler(cls):
		try:
			with open(utils.get_cache_path('webuiapi.json'), 'r') as f:
				j = json.load(f)
				sampler = ['use same sampler']
				sampler.extend(j.get('sampler_list', []))
				return sampler
		except:
			print(f'An error occured in API server.')
			return []

	@classmethod
	def get_scheduler(cls):
		try:
			with open(utils.get_cache_path('webuiapi.json'), 'r') as f:
				j = json.load(f)
				scheduler = ['use same scheduler']
				scheduler.extend(j.get('scheduler_list', []))
				return scheduler
		except:
			print(f'An error occured in API server.')
			return []

	@classmethod
	def get_checkpoint(cls):
		try:
			with open(utils.get_cache_path('webuiapi.json'), 'r') as f:
				j = json.load(f)
				checkpoints = ['use same checkpoint']
				checkpoints.extend(j.get('sd-model_list', []))
				return checkpoints
		except:
			print(f'An error occured in API server.')
			return []

	@classmethod
	def get_controlnet_models(cls):
		try:
			with open(utils.get_cache_path('controlnetapi.json'), 'r') as f:
				j = json.load(f)
				return j.get('models', [])
		except:
			print(f'An error occured in API server.')
			return []

	@classmethod
	def get_controlnet_modules(cls):
		try:
			with open(utils.get_cache_path('controlnetapi.json'), 'r') as f:
				j = json.load(f)
				return j.get('modules', [])
		except:
			print(f'An error occured in API server.')
			return []

	def get_model(self):
		j = self.get('sdapi/v1/options')
		return j['sd_model_checkpoint']

	def change_model(self, name):
		data = {'sd_model_checkpoint': name}
		self.post('sdapi/v1/options', data=json.dumps(data))

	def get_progress(self):
		return self.get('sdapi/v1/progress')


class BMABApiServer:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'server_ip_address': ('STRING', {'default': '127.0.0.1', 'multiline': False}),
				'server_port': ('INT', {'default': 7860, 'min': 1, 'max': 65535, 'step': 1}),
				'checkpoint': (ApiServer.get_checkpoint(),),
			}
		}

	RETURN_TYPES = ('API Server',)
	RETURN_NAMES = ('server',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, server_ip_address, server_port, checkpoint):
		api_server = ApiServer(server_ip_address, server_port)
		api_server.get_all_info()
		if checkpoint != 'use same checkpoint':
			server_checkpoint = api_server.get_model()
			if checkpoint != server_checkpoint:
				api_server.change_model(checkpoint)
		return (api_server,)


class BMABApiSDWebUIT2I:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'api_server': ('API Server',),
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'negative_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'width': ('INT', {'default': 512, 'min': 128, 'max': 4000, 'step': 1}),
				'height': ('INT', {'default': 512, 'min': 128, 'max': 4000, 'step': 1}),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 1000, 'step': 1}),
				'cfg_scale': ('INT', {'default': 7, 'min': 0, 'max': 20, 'step': 0.1}),
				'seed': ('INT', {'default': -1, 'min': -1, 'max': 1000000, 'step': 1}),
			},
			'optional': {
				'hires_fix': ('Hires.fix',),
				'extension': ('EXTENSION',),
				'controlnet': ('ControlNet',),
			}
		}

	RETURN_TYPES = ('API Server', 'IMAGE',)
	RETURN_NAMES = ('API Server', 'image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, api_server: ApiServer, prompt, negative_prompt, width, height, steps, cfg_scale, seed, hires_fix=None, extension=None, controlnet=None):

		txt2img = {
			'prompt': prompt,
			'negative_prompt': negative_prompt,
			'steps': steps,
			'width': width,
			'height': height,
			'cfg_scale': cfg_scale,
			'seed': seed,
			'batch_size': 1,
			'sampler_name': 'DPM++ SDE',
			'scheduler': 'Karras',
			'denoising_strength': 0.5,
			'script_name': None,
			'script_args': [],
			'alwayson_scripts': {}
		}

		if hires_fix is not None:
			txt2img.update(hires_fix)

		if extension is not None:
			txt2img['alwayson_scripts'].update(extension)

		if controlnet is not None:
			txt2img['alwayson_scripts'].update(controlnet)

		# print(json.dumps(txt2img, indent=2))

		def _call_func(c, data):
			res = api_server.post('sdapi/v1/txt2img', data)
			try:
				j = res.json()
				images = j['images']
				image = b64_decoding(images[0])
				c.image = image
			except:
				print(res)
			print('done')

		self.image = None
		th = threading.Thread(target=_call_func, args=(self, txt2img))
		th.start()
		pbar = comfy.utils.ProgressBar(steps)

		while(self.image is None):
			j = api_server.get_progress()
			prog = int(j['progress'] * 100)
			time.sleep(0.5)
			pbar.update_absolute(prog, 100)

		th.join()
		# image = Image.new('RGB', (512, 512))
		pixels = utils.pil2tensor(self.image.convert('RGB'))
		return (api_server, pixels,)


class BMABApiSDWebUII2I:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'api_server': ('API Server',),
				'image': ('IMAGE',),
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'negative_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 1000, 'step': 1}),
				'cfg_scale': ('INT', {'default': 7, 'min': 0, 'max': 20, 'step': 0.1}),
				'seed': ('INT', {'default': -1, 'min': -1, 'max': 1000000, 'step': 1}),
				'sampler': (ApiServer.get_sampler(),),
				'scheduler': (ApiServer.get_scheduler(),),
				'checkpoint': (ApiServer.get_checkpoint(),),
			},
			'optional': {
				'extension': ('EXTENSION',),
				'controlnet': ('ControlNet',),
			}
		}

	RETURN_TYPES = ('API Server', 'IMAGE',)
	RETURN_NAMES = ('API Server', 'image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, api_server: ApiServer, image, prompt, negative_prompt, steps, cfg_scale, seed, sampler, scheduler, checkpoint, extension=None, controlnet=None):
		pixels = image
		image = utils.tensor2pil(pixels)

		img2img = {
			'init_images': [b64_encoding(image)],
			'prompt': prompt,
			'negative_prompt': negative_prompt,
			'steps': steps,
			'width': image.width,
			'height': image.height,
			'cfg_scale': cfg_scale,
			'seed': seed,
			'batch_size': 1,
			'denoising_strength': 0.5,
			'hr_second_pass_steps': steps,
			'resize_mode': 0,
			'script_name': None,
			'script_args': [],
			'alwayson_scripts': {}
		}

		if extension is not None:
			img2img['alwayson_scripts'].update(extension)

		if controlnet is not None:
			img2img['alwayson_scripts'].update(controlnet)

		if sampler != 'use same sampler':
			img2img['sampler_name'] = sampler
		if scheduler != 'use same scheduler':
			img2img['scheduler'] = scheduler
		# if checkpoint != 'use same checkpoint':
		# 	img2img['hr_checkpoint_name'] = checkpoint

		# print(json.dumps(txt2img, indent=2))

		def _call_func(c, data):
			res = api_server.post('sdapi/v1/img2img', data)
			try:
				j = res.json()
				images = j['images']
				image = b64_decoding(images[0])
				c.image = image
			except:
				print(res)
				c.image = 'ERROR'
			print('done')

		self.image = None
		th = threading.Thread(target=_call_func, args=(self, img2img))
		th.start()
		pbar = comfy.utils.ProgressBar(steps)

		while(self.image is None):
			j = api_server.get_progress()
			prog = int(j['progress'] * 100)
			time.sleep(0.5)
			pbar.update_absolute(prog, 100)

		th.join()
		# image = Image.new('RGB', (512, 512))
		if not isinstance(self.image, str):
			pixels = utils.pil2tensor(self.image.convert('RGB'))

		return (api_server, pixels,)


class BMABApiSDWebUIT2IHiresFix:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'negative_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'upscaler': (ApiServer.get_upscaler(),),
				'sampler': (ApiServer.get_sampler(),),
				'scheduler': (ApiServer.get_scheduler(),),
				'checkpoint': (ApiServer.get_checkpoint(),),
				'scale': ('FLOAT', {'default': 2.0, 'min': 0.0, 'max': 4.0, 'step': 0.001}),
				'width': ('INT', {'default': 1024, 'min': 128, 'max': 4000, 'step': 1}),
				'height': ('INT', {'default': 1024, 'min': 128, 'max': 4000, 'step': 1}),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 1000, 'step': 1}),
			}
		}

	RETURN_TYPES = ('Hires.fix',)
	RETURN_NAMES = ('hires_fix',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, prompt, negative_prompt, upscaler, sampler, scheduler, checkpoint, scale, width, height, steps):
		hires_fix = {
			'enable_hr': True,
			'hr_prompt': prompt,
			'hr_negative_prompt': negative_prompt,
			'hr_scale': scale,
			'hr_upscaler': upscaler,
			'hr_resize_x': 0 if scale != 0 else width,
			'hr_resize_y': 0 if scale != 0 else height,
			'hr_second_pass_steps': steps,
		}
		if sampler != 'use same sampler':
			hires_fix['hr_sampler'] = sampler
		if scheduler != 'use same scheduler':
			hires_fix['hr_scheduler'] = scheduler
		if checkpoint != 'use same checkpoint':
			hires_fix['hr_checkpoint_name'] = checkpoint

		return (hires_fix,)


class BMABApiSDWebUIBMABExtension:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'optional': {
				'extension': ('EXTENSION',),
			}
		}

	RETURN_TYPES = ('EXTENSION',)
	RETURN_NAMES = ('extension',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, extension=None):
		if extension is None:
			extension = {
				'BMAB': {
					'args': [{}]
				},
			}

		b = {
			'enabled': True,
			'face_detailing_enabled': True,
			'module_config.controlnet.enabled': True,
			'module_config.controlnet.noise': True,
			'module_config.controlnet.noise_strength': 0.4,
			'module_config.controlnet.noise_begin': 0,
			'module_config.controlnet.noise_end': 0.4,
			'module_config.controlnet.noise_hiresfix': 'Low res only',
		}
		extension['BMAB']['args'][0].update(b)
		return (extension,)


class BMABApiSDWebUIControlNet:
	@classmethod
	def INPUT_TYPES(s):
		input_dir = folder_paths.get_input_directory()
		files = ['None']
		files.extend(sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]))

		return {
			'required': {
				'model': (ApiServer.get_controlnet_models(),),
				'module': (ApiServer.get_controlnet_modules(),),
				'weight': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'start': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'end': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'mode': (('Balanced', 'My prompt is more important', 'ControlNet is more important'),),
				'resize_mode': (('Crop and Resize', 'Just Resize', 'Resize and Fill',),),
				'image': (files, {'image_upload': True}),
			},
			'optional': {
				'controlnet': ('ControlNet',),
			}
		}

	RETURN_TYPES = ('ControlNet',)
	RETURN_NAMES = ('controlnet',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, model, module, weight, start, end, mode, resize_mode, image, controlnet=None):
		if controlnet is None:
			controlnet = {
				'ControlNet': {
					'args': []
				},
			}
		else:
			controlnet = copy.deepcopy(controlnet)
		input_dir = folder_paths.get_input_directory()
		image_path = os.path.join(input_dir, image)
		img = Image.open(image_path)
		b = {
			'enabled': True,
			'image': b64_encoding(img),
			'model': model,
			'module': module,
			'weight': weight,
			'starting/ending': (start, end),
			'resize mode': resize_mode,
			'allow preview': False,
			'pixel perfect': False,
			'control mode': mode,
			'processor_res': 512,
			'threshold_a': 0.5,
			'threshold_b': 0.5,
		}
		controlnet['ControlNet']['args'].append(b)
		return (controlnet,)
