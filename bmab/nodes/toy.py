from bmab import utils
from bmab.nodes.binder import BMABBind
from bmab.external.advanced_clip import advanced_encode


class BMABGoogleGemini:

	def __init__(self) -> None:
		super().__init__()
		self.last_prompt = None
		self.last_seed = None

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'bind': ('BMAB bind',),
				'fixed_seed': ('SEED',),
				'text': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'api_key': ('STRING', {'multiline': False}),
				'token_normalization': (['none', 'mean', 'length', 'length+mean'],),
				'weight_interpretation': (['original', 'comfy', 'A1111', 'compel', 'comfy++', 'down_weight'],),
			}
		}

	RETURN_TYPES = ('BMAB bind',)
	RETURN_NAMES = ('bind', )
	FUNCTION = 'prompt'

	CATEGORY = 'BMAB/toy'

	def prompt(self, bind: BMABBind, fixed_seed, text: str, api_key, token_normalization, weight_interpretation):

		bind = bind.copy()

		if self.last_seed is None or (self.last_seed is not None and self.last_seed != fixed_seed):
			import google.generativeai as genai
			genai.configure(api_key=api_key)
			model = genai.GenerativeModel('gemini-pro')

			text = text.strip()
			response = model.generate_content(f'make detailed prompt for stable diffusion using keyword "{text}" about face, pose, clothes, background and colors. only 1 sentence.')
			print(response)
			self.last_seed = fixed_seed
			self.last_prompt = response.text

		print(self.last_prompt)
		bind.prompt += self.last_prompt
		print(bind.prompt)

		bind.clip = bind.clip.clone()
		prompt = utils.parse_prompt(bind.prompt, bind.seed)

		if weight_interpretation == 'original':
			tokens = bind.clip.tokenize(prompt)
			cond, pooled = bind.clip.encode_from_tokens(tokens, return_pooled=True)
			bind.positive = [[cond, {'pooled_output': pooled}]]
		else:
			embeddings_final, pooled = advanced_encode(bind.clip, prompt, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=False)
			bind.positive = [[embeddings_final, {'pooled_output': pooled}]]

		return (bind, )
