import copy


class BMABContext:

	def __init__(self, *args) -> None:
		super().__init__()
		self.seed, self.sampler, self.scheduler, self.cfg_scale, self.steps = args

	def get(self):
		return self.seed, self.sampler, self.scheduler, self.cfg_scale, self.steps

	def update(self, steps, cfg_scale, sampler, scheduler):
		if steps == 0:
			steps = self.steps
		if cfg_scale == 0:
			cfg_scale = self.cfg_scale
		if sampler != 'Use same sampler':
			sampler = self.sampler
		if scheduler != 'Use same scheduler':
			scheduler = self.scheduler
		return steps, cfg_scale, sampler, scheduler


class BMABBind:

	def __init__(self, *args) -> None:
		super().__init__()

		self.model, self.clip, self.vae, self.prompt, self.negative_prompt, self.positive, self.negative, self.latent_image, self.context, self.pixels, self.seed = args

	def copy(self):
		return copy.copy(self)

	@staticmethod
	def result(bind, pixels, *args):
		if bind is None:
			return (None, pixels, *args)
		else:
			bind.pixels = pixels
			return (bind, bind.pixels, *args)

	def get(self):
		return self.model, self.clip, self.vae, self.prompt, self.negative_prompt, self.positive, self.negative, self.latent_image, self.context, self.pixels, self.seed


class BMABLoraBind:
	def __init__(self, *args) -> None:
		super().__init__()
		self.loras = []

	def append(self, *args):
		self.loras.append(args)
