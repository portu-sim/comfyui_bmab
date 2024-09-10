import torch
import comfy
import nodes
import folder_paths
import node_helpers

from PIL import Image

from comfy_extras.chainner_models import model_loading
from comfy import model_management
from comfy_extras import nodes_clip_sdxl

from bmab import utils
from bmab.nodes.binder import BMABBind
from bmab.nodes.binder import BMABContext
from bmab.nodes.binder import BMABLoraBind
from bmab.external.advanced_clip import advanced_encode


class BMABContextNode:

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
				'steps': ('INT', {'default': 20, 'min': 1, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (comfy.samplers.KSampler.SCHEDULERS,),
			},
			'optional': {
				'seed_in': ('SEED',),
			}
		}

	RETURN_TYPES = ('CONTEXT', )
	RETURN_NAMES = ('BMAB context', )
	FUNCTION = 'process'

	CATEGORY = 'BMAB/sampler'

	def process(self, seed, steps, cfg_scale, sampler_name, scheduler, seed_in=None):
		if seed_in is not None:
			seed = seed_in
		context = BMABContext(seed, steps, cfg_scale, sampler_name, scheduler)
		return (context, )


class BMABIntegrator:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'model': ('MODEL',),
				'clip': ('CLIP',),
				'vae': ('VAE',),
				'context': ('CONTEXT',),
				'stop_at_clip_layer': ('INT', {'default': -2, 'min': -24, 'max': -1, 'step': 1}),
				'token_normalization': (['none', 'mean', 'length', 'length+mean'],),
				'weight_interpretation': (['original', 'sdxl', 'comfy', 'A1111', 'compel', 'comfy++', 'down_weight'],),

				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'negative_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
			},
			'optional': {
				'seed_in': ('SEED',),
				'latent': ('LATENT',),
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind', )
	RETURN_NAMES = ('BMAB bind', )
	FUNCTION = 'integrate_inputs'

	CATEGORY = 'BMAB/sampler'

	def encode_sdxl(self, clip, text_g, text_l):
		width, height, crop_w, crop_h, target_width, target_height = 1024, 1024, 0, 0, 1024, 1024
		tokens = clip.tokenize(text_g)
		tokens['l'] = clip.tokenize(text_l)['l']
		if len(tokens['l']) != len(tokens['g']):
			empty = clip.tokenize('')
			while len(tokens['l']) < len(tokens['g']):
				tokens['l'] += empty['l']
			while len(tokens['l']) > len(tokens['g']):
				tokens['g'] += empty['g']
		cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
		return [[cond, {'pooled_output': pooled, 'width': width, 'height': height, 'crop_w': crop_w, 'crop_h': crop_h, 'target_width': target_width, 'target_height': target_height}]]

	def integrate_inputs(self, model, clip, vae, context: BMABContext, stop_at_clip_layer, token_normalization, weight_interpretation, prompt, negative_prompt, seed_in=None, latent=None, image=None):
		if context is None and seed_in is None:
			print('No SEED defined.')
			raise ValueError('No SEED defined')

		if context is None:
			seed = seed_in
		else:
			seed = context.seed

		prompt = utils.parse_prompt(prompt, seed)

		if weight_interpretation == 'original':
			tokens = clip.tokenize(prompt)
			cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
			positive = [[cond, {'pooled_output': pooled}]]
			tokens = clip.tokenize(negative_prompt)
			cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
			negative = [[cond, {'pooled_output': pooled}]]
		elif weight_interpretation == 'sdxl':
			prompt = utils.parse_prompt(prompt, seed)
			negative_prompt = utils.parse_prompt(negative_prompt, seed)
			positive = self.encode_sdxl(clip, prompt, prompt)
			negative = self.encode_sdxl(clip, negative_prompt, negative_prompt)
		else:
			embeddings_final, pooled = advanced_encode(clip, prompt, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=False)
			positive = [[embeddings_final, {'pooled_output': pooled}]]
			embeddings_final, pooled = advanced_encode(clip, negative_prompt, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=False)
			negative = [[embeddings_final, {'pooled_output': pooled}]]

		clip.clip_layer(stop_at_clip_layer)

		return BMABBind(model, clip, vae, prompt, negative_prompt, positive, negative, latent, context, image, seed),



class BMABFluxIntegrator:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'model': ('MODEL',),
				'clip': ('CLIP',),
				'vae': ('VAE',),
				'context': ('CONTEXT',),
				'guidance': ('FLOAT', {'default': 3.5, 'min': 0.0, 'max': 100.0, 'step': 0.1}),
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
			},
			'optional': {
				'seed_in': ('SEED',),
				'latent': ('LATENT',),
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind', )
	RETURN_NAMES = ('BMAB bind', )
	FUNCTION = 'integrate_inputs'

	CATEGORY = 'BMAB/sampler'

	def integrate_inputs(self, model, clip, vae, context: BMABContext, guidance, prompt, seed_in=None, latent=None, image=None):
		if context is None and seed_in is None:
			print('No SEED defined.')
			raise ValueError('No SEED defined')

		if context is None:
			seed = seed_in
		else:
			seed = context.seed

		prompt = utils.parse_prompt(prompt, seed)

		tokens = clip.tokenize(prompt)
		output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
		cond = output.pop("cond")
		positive = [[cond, output]]
		positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

		tokens = clip.tokenize('')
		output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
		cond = output.pop("cond")
		negative = [[cond, output]]

		return BMABBind(model, clip, vae, prompt, '', positive, negative, latent, context, image, seed),


class BMABImportIntegrator:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'model': ('MODEL',),
				'clip': ('CLIP',),
				'vae': ('VAE',),
				'context': ('CONTEXT',),
				'positive': ('CONDITIONING',),
				'negative': ('CONDITIONING',),
				'stop_at_clip_layer': ('INT', {'default': -2, 'min': -24, 'max': -1, 'step': 1}),
			},
			'optional': {
				'seed_in': ('SEED',),
				'latent': ('LATENT',),
				'image': ('IMAGE',),
			}
		}

	RETURN_TYPES = ('BMAB bind', )
	RETURN_NAMES = ('BMAB bind', )
	FUNCTION = 'integrate_inputs'

	CATEGORY = 'BMAB/sampler'

	def integrate_inputs(self, model, clip, vae, context: BMABContext, positive, negative, stop_at_clip_layer, seed_in=None, latent=None, image=None):
		if context is None and seed_in is None:
			print('No SEED defined.')
			raise ValueError('No SEED defined')

		if context is None:
			seed = seed_in
		else:
			seed = context.seed

		clip.clip_layer(stop_at_clip_layer)

		return BMABBind(model, clip, vae, None, None, positive, negative, latent, context, image, seed),


class BMABExtractor:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
			},
		}

	RETURN_TYPES = ('MODEL', 'CLIP', 'CONDITIONING', 'CONDITIONING', 'VAE', 'LATENT', 'IMAGE', 'SEED')
	RETURN_NAMES = ('model', 'clip', 'positive', 'negative', 'vae', 'latent', 'image', 'seed')
	FUNCTION = 'extract'

	CATEGORY = 'BMAB/sampler'

	def extract(self, bind: BMABBind):
		bind = bind.copy()
		if bind.pixels is not None:
			t = bind.vae.encode(bind.pixels)
			bind.latent_image = {'samples': t}
		return bind.model, bind.clip, bind.positive, bind.negative, bind.vae, bind.latent_image, bind.pixels, bind.seed,


class BMABSeedGenerator:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
			}
		}

	RETURN_TYPES = ('SEED',)
	RETURN_NAMES = ('seed',)
	FUNCTION = 'generate'

	CATEGORY = 'BMAB/sampler'

	def generate(self, seed):
		return seed,


class BMABKSampler:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
			},
			'optional': {
				'lora': ('BMAB lora',)
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE',)
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = 'sample'

	CATEGORY = 'BMAB/sampler'

	def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
		print(f'Loading lora {lora_name}')
		lora_path = folder_paths.get_full_path('loras', lora_name)
		lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
		model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
		return (model_lora, clip_lora)

	def sample(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, lora: BMABLoraBind = None):
		print('Sampler SEED', bind.seed)
		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)
		if lora is not None:
			for l in lora.loras:
				bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg_scale, sampler_name, scheduler, bind.positive, bind.negative, bind.latent_image, denoise=denoise)[0]
		bind.pixels = bind.vae.decode(samples['samples'])
		return bind, bind.pixels,


class BMABKSamplerHiresFix:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 0.4, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
			},
			'optional': {
				'image': ('IMAGE',),
				'lora': ('BMAB lora',)
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE',)
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = 'sample'

	CATEGORY = 'BMAB/sampler'

	def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
		print(f'Loading lora {lora_name}')
		lora_path = folder_paths.get_full_path('loras', lora_name)
		lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
		model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
		return (model_lora, clip_lora)

	def sample(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise=1.0, image=None, lora: BMABLoraBind = None):
		pixels = bind.pixels if image is None else image
		if pixels is None:
			pixels = bind.vae.decode(bind.latent_image['samples'])

		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)
		print('Hires SEED', bind.seed, bind.model)
		latent = dict(samples=bind.vae.encode(pixels))
		if lora is not None:
			for l in lora.loras:
				bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg_scale, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise, force_full_denoise=True)[0]
		bind.pixels = bind.vae.decode(samples['samples'])
		return bind, bind.pixels,


class BMABKSamplerHiresFixWithUpscaler:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 0.4, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'model_name': (['LANCZOS'] + folder_paths.get_filename_list('upscale_models'),),
				'scale': ('FLOAT', {'default': 2.0, 'min': 0, 'max': 4.0, 'step': 0.001}),
				'width': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
			},
			'optional': {
				'image': ('IMAGE',),
				'lora': ('BMAB lora',)
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE',)
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = 'sample'

	CATEGORY = 'BMAB/sampler'

	def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
		print(f'Loading lora {lora_name}')
		lora_path = folder_paths.get_full_path('loras', lora_name)
		lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
		model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
		return (model_lora, clip_lora)

	def load_model(self, model_name):
		model_path = folder_paths.get_full_path('upscale_models', model_name)
		sd = comfy.utils.load_torch_file(model_path, safe_load=True)
		if 'module.layers.0.residual_group.blocks.0.norm1.weight' in sd:
			sd = comfy.utils.state_dict_prefix_replace(sd, {'module.': ''})
		out = model_loading.load_state_dict(sd).eval()
		return out

	def upscale_with_model(self, model_name, pixels):
		upscale_model = self.load_model(model_name)
		device = model_management.get_torch_device()

		memory_required = model_management.module_size(upscale_model.model)
		memory_required += (512 * 512 * 3) * pixels.element_size() * max(upscale_model.scale, 1.0) * 384.0  # The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
		memory_required += pixels.nelement() * pixels.element_size()
		model_management.free_memory(memory_required, device)

		upscale_model.to(device)
		in_img = pixels.movedim(-1, -3).to(device)

		tile = 512
		overlap = 32

		oom = True
		while oom:
			try:
				steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
				pbar = comfy.utils.ProgressBar(steps)
				s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
				oom = False
			except model_management.OOM_EXCEPTION as e:
				tile //= 2
				if tile < 128:
					raise e

		upscale_model.to('cpu')
		s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
		return (s,)

	def sample(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, model_name, scale, width, height,  image=None, lora: BMABLoraBind = None):
		pixels = bind.pixels if image is None else image
		if pixels is None:
			pixels = bind.vae.decode(bind.latent_image['samples'])

		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)
		if scale != 0:
			_, h, w, c = pixels.shape
			width, height = int(w * scale), int(h * scale)

		if scale != 1:
			if model_name == 'LANCZOS':
				pil_images = utils.get_pils_from_pixels(pixels)
				results = [img.resize((width, height), Image.Resampling.LANCZOS) for img in pil_images]
				pixels = utils.get_pixels_from_pils(results)
			else:
				if scale > 1:
					s = self.upscale_with_model(model_name, pixels)
				pil_images = utils.get_pils_from_pixels(s)
				results = [img.resize((width, height), Image.Resampling.LANCZOS) for img in pil_images]
				pixels = utils.get_pixels_from_pils(results)

		print('Hires SEED', bind.seed, bind.model)
		latent = dict(samples=bind.vae.encode(pixels))
		if lora is not None:
			for l in lora.loras:
				bind.model, bind.clip = self.load_lora(bind.model, bind.clip, *l)
		samples = nodes.common_ksampler(bind.model, bind.seed, steps, cfg_scale, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise, force_full_denoise=True)[0]
		bind.pixels = bind.vae.decode(samples['samples'])
		return bind, bind.pixels,


class BMABPrompt:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'text': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'token_normalization': (['none', 'mean', 'length', 'length+mean'],),
				'weight_interpretation': (['original', 'sdxl', 'comfy', 'A1111', 'compel', 'comfy++', 'down_weight'],),
			}
		}

	RETURN_TYPES = ('BMAB bind',)
	RETURN_NAMES = ('bind', )
	FUNCTION = 'prompt'

	CATEGORY = 'BMAB/sampler'

	def encode_sdxl(self, clip, text_g, text_l):
		width, height, crop_w, crop_h, target_width, target_height = 1024, 1024, 0, 0, 1024, 1024
		tokens = clip.tokenize(text_g)
		tokens['l'] = clip.tokenize(text_l)['l']
		if len(tokens['l']) != len(tokens['g']):
			empty = clip.tokenize('')
			while len(tokens['l']) < len(tokens['g']):
				tokens['l'] += empty['l']
			while len(tokens['l']) > len(tokens['g']):
				tokens['g'] += empty['g']
		cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
		return [[cond, {'pooled_output': pooled, 'width': width, 'height': height, 'crop_w': crop_w, 'crop_h': crop_h, 'target_width': target_width, 'target_height': target_height}]]

	def prompt(self, bind: BMABBind, text, token_normalization, weight_interpretation):

		bind = bind.copy()
		if text.find('__prompt__') >= 0 and bind.prompt is not None and bind.prompt != '':
			text = text.replace('__prompt__', bind.prompt)
		bind.prompt = text
		bind.clip = bind.clip.clone()
		prompt = utils.parse_prompt(bind.prompt, bind.seed)

		if weight_interpretation == 'original':
			tokens = bind.clip.tokenize(prompt)
			cond, pooled = bind.clip.encode_from_tokens(tokens, return_pooled=True)
			bind.positive = [[cond, {'pooled_output': pooled}]]
		elif weight_interpretation == 'sdxl':
			prompt = utils.parse_prompt(text, bind.seed)
			bind.positive = self.encode_sdxl(bind.clip, prompt, prompt)
		else:
			embeddings_final, pooled = advanced_encode(bind.clip, prompt, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=False)
			bind.positive = [[embeddings_final, {'pooled_output': pooled}]]

		return (bind, )


class BMABKSamplerKohyaDeepShrink:
	upscale_methods = ['bicubic', 'nearest-exact', 'bilinear', 'area', 'bislerp']

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'steps': ('INT', {'default': 20, 'min': 0, 'max': 10000}),
				'cfg_scale': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
				'sampler_name': (['Use same sampler'] + comfy.samplers.KSampler.SAMPLERS,),
				'scheduler': (['Use same scheduler'] + comfy.samplers.KSampler.SCHEDULERS,),
				'denoise': ('FLOAT', {'default': 0.4, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
				'scale': ('FLOAT', {'default': 2.0, 'min': 0, 'max': 4.0, 'step': 0.001}),
				'width': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
				'height': ('INT', {'default': 512, 'min': 0, 'max': nodes.MAX_RESOLUTION, 'step': 8}),
				'block_number': ('INT', {'default': 3, 'min': 1, 'max': 32, 'step': 1}),
				'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
				'end_percent': ('FLOAT', {'default': 0.35, 'min': 0.0, 'max': 1.0, 'step': 0.001}),
				'downscale_after_skip': ('BOOLEAN', {'default': True}),
				'downscale_method': (s.upscale_methods,),
				'upscale_method': (s.upscale_methods,),
			},
			'optional': {
				'image': ('IMAGE',),
				'lora': ('BMAB lora',)
			}
		}

	RETURN_TYPES = ('BMAB bind', 'IMAGE',)
	RETURN_NAMES = ('BMAB bind', 'image',)
	FUNCTION = 'sample'

	CATEGORY = 'BMAB/sampler'

	def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
		print(f'Loading lora {lora_name}')
		lora_path = folder_paths.get_full_path('loras', lora_name)
		lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
		model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
		return (model_lora, clip_lora)

	def sample(self, bind: BMABBind, steps, cfg_scale, sampler_name, scheduler, denoise, scale, width, height, block_number, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, image=None, lora: BMABLoraBind = None):
		pixels = bind.pixels if image is None else image
		if pixels is None:
			pixels = bind.vae.decode(bind.latent_image['samples'])

		if bind.context is not None:
			steps, cfg_scale, sampler_name, scheduler = bind.context.update(steps, cfg_scale, sampler_name, scheduler)
		if scale != 0:
			_, h, w, c = pixels.shape
			width, height = int(w * scale), int(h * scale)
		_, org_height, org_width, c = pixels.shape
		_scale = height / org_height
		r_length = max(width, height)
		downscale_factor = r_length / 1024
		print('downscale_factor', downscale_factor)

		m = self.patch(bind.model, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method)

		pil_images = utils.get_pils_from_pixels(pixels)
		results = [utils.resize_and_fill(img, r_length, r_length) for img in pil_images]
		pixels = utils.get_pixels_from_pils(results)

		print('Hires SEED', bind.seed, bind.model)
		latent = dict(samples=bind.vae.encode(pixels))
		if lora is not None:
			for l in lora.loras:
				m, bind.clip = self.load_lora(m, bind.clip, *l)
		samples = nodes.common_ksampler(m, bind.seed, steps, cfg_scale, sampler_name, scheduler, bind.positive, bind.negative, latent, denoise=denoise, force_full_denoise=True)[0]
		pixels = bind.vae.decode(samples['samples'])
		pil_images = utils.get_pils_from_pixels(pixels)
		results = [utils.crop(img, width, height) for img in pil_images]
		bind.pixels = utils.get_pixels_from_pils(results)
		return bind, bind.pixels,

	def patch(self, model, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method):
		model_sampling = model.get_model_object('model_sampling')
		sigma_start = model_sampling.percent_to_sigma(start_percent)
		sigma_end = model_sampling.percent_to_sigma(end_percent)

		def input_block_patch(h, transformer_options):
			if transformer_options['block'][1] == block_number:
				sigma = transformer_options['sigmas'][0].item()
				if sigma <= sigma_start and sigma >= sigma_end:
					h = comfy.utils.common_upscale(h, round(h.shape[-1] * (1.0 / downscale_factor)), round(h.shape[-2] * (1.0 / downscale_factor)), downscale_method, 'disabled')
			return h

		def output_block_patch(h, hsp, transformer_options):
			if h.shape[2] != hsp.shape[2]:
				h = comfy.utils.common_upscale(h, hsp.shape[-1], hsp.shape[-2], upscale_method, 'disabled')
			return h, hsp

		m = model.clone()
		if downscale_after_skip:
			m.set_model_input_block_patch_after_skip(input_block_patch)
		else:
			m.set_model_input_block_patch(input_block_patch)
		m.set_model_output_block_patch(output_block_patch)
		return m


class BMABClipTextEncoderSDXL(nodes_clip_sdxl.CLIPTextEncodeSDXL):

	@classmethod
	def INPUT_TYPES(s):
		dic = super().INPUT_TYPES()
		dic['optional'] = {
			'seed': ('SEED', )
		}
		return dic

	CATEGORY = 'BMAB/sampler'

	def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, seed=None):
		if seed is not None:
			text_g = utils.parse_prompt(text_g, seed)
			text_l = utils.parse_prompt(text_l, seed)
		return super().encode(clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l)


class BMABToBind:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'context': ('CONTEXT',),
			},
			'optional': {
				'bind': ('BMAB bind',),
				'model': ('MODEL',),
				'clip': ('CLIP',),
				'vae': ('VAE',),
				'positive': ('CONDITIONING',),
				'negative': ('CONDITIONING',),
				'latent': ('LATENT',),
			}
		}

	RETURN_TYPES = ('BMAB bind', )
	RETURN_NAMES = ('BMAB bind', )
	FUNCTION = 'integrate_inputs'

	CATEGORY = 'BMAB/sampler'

	def integrate_inputs(self, context: BMABContext, model=None, clip=None, vae=None, positive=None, negative=None, bind=None, latent=None):
		if bind is not None:
			if model is not None:
				bind.model = model
			if clip is not None:
				bind.clip = clip
			if vae is not None:
				bind.vae = vae
			if positive is not None:
				bind.positive = positive
			if negative is not None:
				bind.negative = negative
			if latent is not None:
				bind.latent = latent
			return bind
		else:
			return BMABBind(model, clip, vae, '', '', positive, negative, latent, context, None, context.seed),
