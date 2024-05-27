import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from bmab import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
	import testnodes
	print('Register test nodes.')
	testnodes.register(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
except:
	print('Not found test nodes.')

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = f'./web'

