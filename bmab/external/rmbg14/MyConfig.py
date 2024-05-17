from transformers import PretrainedConfig
from typing import List

class RMBGConfig(PretrainedConfig):
    model_type = "SegformerForSemanticSegmentation"
    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        **kwargs):
      self.in_ch = in_ch
      self.out_ch = out_ch
      super().__init__(**kwargs)
