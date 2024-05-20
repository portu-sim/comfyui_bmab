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

	CATEGORY = 'BMAB/sampler'

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

	CATEGORY = 'BMAB/sampler'

	def process(self, bind: BMABBind, positive=None, negative=None):
		if positive is not None:
			bind.positive = positive
		if negative is not None:
			bind.negative = negative
		return (bind, )

