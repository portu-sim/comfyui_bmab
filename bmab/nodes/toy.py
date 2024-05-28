import os
import random
import time
import bmab

from bmab import utils


question = '''
make detailed prompt for stable diffusion using keyword "{text}" about scene, lighting, face, pose, clothes, background and colors in only 1 sentence. The sentence is describes very detailed. Do not say about human race.
'''


class BMABGoogleGemini:

	def __init__(self) -> None:
		super().__init__()
		self.last_prompt = None
		self.last_text = None

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'text': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
				'api_key': ('STRING', {'multiline': False}),
				'random_seed': ('INT', {'default': 0, 'min': 0, 'max': 65536, 'step': 1}),
			},
		}

	RETURN_TYPES = ('STRING',)
	RETURN_NAMES = ('string', )
	FUNCTION = 'prompt'

	CATEGORY = 'BMAB/toy'

	def get_prompt(self, text, api_key):
		import google.generativeai as genai
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel('gemini-pro')
		text = text.strip()
		response = model.generate_content(question.format(text=text))
		try:
			self.last_prompt = response.text
			print(response.text)
			cache_path = os.path.join(os.path.dirname(bmab.__file__), '../resources/cache')
			cache_file = os.path.join(cache_path, 'gemini.txt')
			with open(cache_file, 'a', encoding='utf8') as f:
				f.write(time.strftime('%Y.%m.%d - %H:%M:%S'))
				f.write('\n')
				f.write(self.last_prompt)
				f.write('\n')
		except:
			print('ERROR reading API response', response)
		return self.last_prompt

	def prompt(self, prompt: str, text: str, api_key, random_seed=None, **kwargs):
		random_seed = random.randint(0, 65535)
		if prompt.find('__prompt__') >= 0:
			if self.last_text != text:
				random_seed = random.randint(0, 65535)
				self.last_text = text
				self.get_prompt(text, api_key)
			prompt = prompt.replace('__prompt__', self.last_prompt)
		result = utils.parse_prompt(prompt, random_seed)
		return {"ui": {"string": [str(random_seed), ]}, "result": (result,)}
