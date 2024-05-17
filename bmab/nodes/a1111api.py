import json
import requests

from PIL import Image
from bmab import utils
from bmab.nodes.binder import BMABBind

import base64
from io import BytesIO


def b64_encoding(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def b64_decoding(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))


class BMABApiBase:
	pass


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
		return resp.json()

	def get_all_info(self):
		resources = [
			'samplers',
			'schedulers',
			'upscalers',
			'sd-models',
			'sd-vae',
			'loras',
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
		self.controlnet_info['modules'] = self.get(f'controlnet/model_list').get('module_list', [])

		with open(utils.get_cache_path('webuiapi.json'), 'w') as f:
			json.dump(self.sdwebui_info, f)

		with open(utils.get_cache_path('controlnetapi.json'), 'w') as f:
			json.dump(self.controlnet_info, f)


class BMABApiServer:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'server_ip_address': ('STRING', {'default': '127.0.0.1', 'multiline': False}),
				'server_port': ('INT', {'default': 7860, 'min': 1, 'max': 65535, 'step': 1}),
			}
		}

	RETURN_TYPES = ('API Server', )
	RETURN_NAMES = ('server', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, server_ip_address, server_port):
		api_server = ApiServer(server_ip_address, server_port)
		api_server.get_all_info()
		return (api_server, )


class BMABApiSDWebUIT2I(BMABApiBase):
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'api_server': ('API Server', ),
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'negative_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'width': ('INT', {'default': 512, 'min': 128, 'max': 4000, 'step': 1}),
				'height': ('INT', {'default': 512, 'min': 128, 'max': 4000, 'step': 1}),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 1000, 'step': 1}),
				'cfg_scale': ('INT', {'default': 7, 'min': 0, 'max': 20, 'step': 0.1}),
				'seed': ('INT', {'default': -1, 'min': -1, 'max': 1000000, 'step': 1}),
			},
			'optional': {
				'hires_fix': ('Hires.fix', ),
				'extension': ('EXTENSION', ),
			}
		}

	RETURN_TYPES = ('API Server', 'IMAGE', )
	RETURN_NAMES = ('API Server', 'image', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, api_server: ApiServer, prompt, negative_prompt, width, height, steps, cfg_scale, seed, hires_fix=None, extension=None):

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

		print(json.dumps(txt2img, indent=2))

		res = api_server.post('sdapi/v1/txt2img', txt2img)
		try:
			j = res.json()

			images = j['images']
			image = b64_decoding(images[0])
		except:
			print(res)

		# image = Image.new('RGB', (512, 512))
		pixels = utils.pil2tensor(image.convert('RGB'))

		return (api_server, pixels, )


class BMABApiSDWebUIT2IHiresFix:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
			}
		}

	RETURN_TYPES = ('Hires.fix',)
	RETURN_NAMES = ('hires_fix',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self):
		hires_fix = {
			'enable_hr': True,
			'hr_scale': 2,
			'hr_upscaler': '4x-UltraSharp',
		}
		return (hires_fix, )


class BMABApiSDWebUIBMABExtension:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'extension': ('EXTENSION',),
			}
		}

	RETURN_TYPES = ('EXTENSION', )
	RETURN_NAMES = ('extension', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self):
		bmab_extension = {
			'BMAB': {
				'args': [
					{
						'enabled': True,
						'face_detailing_enabled': True,
						'module_config.controlnet.enabled': True,
						'module_config.controlnet.noise': True,
						'module_config.controlnet.noise_strength': 0.4,
						'module_config.controlnet.noise_begin': 0,
						'module_config.controlnet.noise_end': 0.4,
						'module_config.controlnet.noise_hiresfix': 'Low res only',
					}
				]
			},
		}
		return (bmab_extension, )



class BMABApiSDWebUIControlNet:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'api_server': ('API Server',),
			}
		}

	RETURN_TYPES = ('API Server', 'IMAGE',)
	RETURN_NAMES = ('API Server', 'image',)
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sdwebui'

	def process(self, api_server):
		return (api_server, )

	def controlnet_struct(self):
		controlnet = {
			'ControlNet': {
				'args': [
					{
						'input_image': b64_encoding(poseimage),
						'model': 'control_sd15_openpose [fef5e48e]',
						'module': 'openpose',
						'weight': 1,
						'starting/ending': (0, 1),
						'resize mode': 'Just Resize',
						'allow preview': False,
						'pixel perfect': False,
						'control mode': 'My prompt is more important',
						'processor_res': 512,
						'threshold_a': 64,
						'threshold_b': 64,
					}
				]
			},
		}
