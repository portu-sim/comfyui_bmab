import os
import sys
import pillow_avif

sys.path.append(os.path.join(os.path.dirname(__file__)))
from bmab import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
	import testnodes
	print('Register test nodes.')
	testnodes.register(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
	print('Not found test nodes.')
	print(e)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = f'./web'

