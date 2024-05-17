import folder_paths
from bmab.nodes.binder import BMABLoraBind


class BMABLoraLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'lora_name': (folder_paths.get_filename_list('loras'), ),
				'strength_model': ('FLOAT', {'default': 1.0, 'min': -100.0, 'max': 100.0, 'step': 0.01}),
				'strength_clip': ('FLOAT', {'default': 1.0, 'min': -100.0, 'max': 100.0, 'step': 0.01}),
			},
			'optional': {
				'lora': ('BMAB lora',),
			}
		}

	RETURN_TYPES = ('BMAB lora', )
	RETURN_NAMES = ('lora', )
	FUNCTION = 'load_lora'

	CATEGORY = 'BMAB/loader'

	def load_lora(self, lora_name, strength_model, strength_clip, lora: BMABLoraBind=None):
		if lora is None:
			lora = BMABLoraBind()
		lora.append(lora_name, strength_model, strength_clip)
		return (lora, )

