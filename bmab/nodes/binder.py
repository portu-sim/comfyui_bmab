import copy


class BMABBind:

	def __init__(self, *args) -> None:
		super().__init__()
		self.model, self.clip, self.vae, self.prompt, self.negative_prompt, self.positive, self.negative, self.latent_image, self.seed, self.pixels = args

	def copy(self):
		return copy.copy(self)

	@staticmethod
	def result(bind, pixels, *args):
		if bind is None:
			return (None, pixels, *args)
		else:
			bind.pixels = pixels
			return (bind, bind.pixels, *args)


class BMABLoraBind:
	def __init__(self, *args) -> None:
		super().__init__()
		self.loras = []

	def append(self, *args):
		self.loras.append(args)

