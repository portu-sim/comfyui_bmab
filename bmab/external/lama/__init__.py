# https://github.com/advimman/lama
# https://github.com/Mikubill/sd-webui-controlnet

import os
import gc
import sys
import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange

from bmab.external.lama.saicinpainting.training.trainers import load_checkpoint
from bmab import utils


def lama_inpainting(image, mask, device='gpu'):
    mask_input = mask
    lama = LamaInpainting()
    if device == 'cpu':
        lama.device = 'cpu'

    if device == 'mps':
        image = lama(image, mask_input)
    elif lama.device.startswith('cuda') and image.width != image.height:
        width, height = image.size
        mx = max(image.width, image.height)
        resized = Image.new('RGB', (mx, mx), 0)
        mask = Image.new('L', (mx, mx), 0)
        if height < width:
            y0 = (mx - height) // 2
            resized.paste(image, (0, y0))
            mask.paste(mask_input, (0, y0))
            l = lama(resized, mask)
            image = l.crop((0, y0, width, y0 + height))
        else:
            x0 = (mx - width) // 2
            resized.paste(image, (x0, 0))
            mask.paste(mask_input, (x0, 0))
            l = lama(resized, mask)
            image = l.crop((x0, 0, x0 + width, height))
    else:
        image = lama(image, mask_input)
    del lama
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return image


class LamaInpainting:

    def __init__(self):
        if sys.platform == 'darwin':
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = None

    @staticmethod
    def load_image(pilimg, mode='RGB'):
        img = np.array(pilimg.convert(mode))
        if img.ndim == 3:
            print('transpose')
            img = np.transpose(img, (2, 0, 1))
        out_img = img.astype('float32') / 255
        return out_img

    def load_model(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        cfg = yaml.safe_load(open(config_path, 'rt'))
        cfg = OmegaConf.create(cfg)
        cfg.training_model.predict_only = True
        cfg.visualizer.kind = 'noop'
        lamapth = utils.lazy_loader('ControlNetLama.pth')
        self.model = load_checkpoint(cfg, lamapth, strict=False, map_location='cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, image, mask):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)

        opencv_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)[:, :, 0:3]
        opencv_mask = cv2.cvtColor(np.array(mask.convert('RGB')), cv2.COLOR_RGB2BGR)[:, :, 0:1]
        color = np.ascontiguousarray(opencv_image).astype(np.float32) / 255.0
        mask = np.ascontiguousarray(opencv_mask).astype(np.float32) / 255.0

        with torch.no_grad():
            color = torch.from_numpy(color).float().to(self.device)
            mask = torch.from_numpy(mask).float().to(self.device)
            mask = (mask > 0.5).float()
            color = color * (1 - mask)
            image_feed = torch.cat([color, mask], dim=2)
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')
            result = self.model(image_feed)[0]
            result = rearrange(result, 'c h w -> h w c')
            result = result * mask + color * (1 - mask)
            result *= 255.0

            img = result.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)
            return pil_image
