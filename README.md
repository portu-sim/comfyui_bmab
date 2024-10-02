# comfyui_bmab

BMAB is an custom nodes of ComfyUI and has the function of post-processing the generated image according to settings.
If necessary, you can find and redraw people, faces, and hands, or perform functions such as resize, resample, and add noise.
You can composite two images or perform the Upscale function.

<img src="https://i.ibb.co/341r93k/2024-05-21-10-56-02.png"/>

You can download sample.json.

https://github.com/portu-sim/comfyui_bmab/blob/main/resources/examples/example.json

# Flux

BMAB now supports Flux 1.


https://github.com/portu-sim/comfyui_bmab/blob/main/resources/examples/bmab-flux-sample.json

<img src="https://github.com/user-attachments/assets/dfa36eb1-6978-47e7-a0d0-67c82639cf52"/>

# New Nodes

## BMAB Inpaint

<img width="1393" alt="image" src="https://github.com/user-attachments/assets/8bf15832-7be4-41f2-b8ac-64056d5ac4d9">

## BMAB Reframe

<img width="1317" alt="image" src="https://github.com/user-attachments/assets/ba971f72-abfa-4998-a6cb-fcbaa245fcb5">

## BMAB Outpaint By Ratio

<img width="1186" alt="image" src="https://github.com/user-attachments/assets/88255a25-6e56-427c-9683-6ea1f591e4dd">



### Gallery

[instagram](https://www.instagram.com/portu.sim/)   
[facebook](https://www.facebook.com/portusimkr)

### Hand Detailing Sample

BMAB detects and enlarges the upper body of a person and performs Openpose at high resolution to fix incorrectly drawn hands.

<img src="https://i.ibb.co/ZMGdXVp/resize-2024-05-23-1-42-12.png"/>


# Installation

You can install comfyui_bmab using ComfyUI-Manager easily.   
You will need to install a total of three custom nodes.

* comfyui_bmab
* comfyui_controlnet_aux
  * https://github.com/Fannovel16/comfyui_controlnet_aux.git
  * Fannovel16, Thanks for excellent code.
* ComfyUI_IPAdapter_plus
  * https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
  * cubiq, Thanks for excellent code.


### Grounding DINO Installation

Transfomer v4.40.0 has Grounding DINO implementation.   
https://github.com/huggingface/transformers/releases/tag/v4.40.0
Now BMAB use transformer for detecting object.
No installation required.

## Install Manually

I can't describe about your python environment.
I will write the installation instructions assuming you have some knowledge of Python.


### Windows portable User

```commandline
cd ComfyUI/custom_nodes
git clone https://github.com/portu-sim/comfyui_bmab.git
cd comfyui_bmab
python_embeded\python.exe -m pip install -r requirements.txt
cd ..
```

You will need to install two additional custom nodes required by comfyui_bmab.

```commandline
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd comfyui_controlnet_aux
python_embeded\python.exe -r pip install requirements.txt
cd ..
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
cd ComfyUI_IPAdapter_plus
python_embeded\python.exe -m pip install -r requirements.txt
cd ..
```

### Other python environment

```commandline
cd ComfyUI/custom_nodes
git clone https://github.com/portu-sim/comfyui_bmab.git
cd comfyui_bmab
pip install -r requirements.txt
cd ..
```

You will need to install two additional custom nodes required by comfyui_bmab.

```commandline
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd comfyui_controlnet_aux
pip install -r requirements.txt
cd ..
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
cd ComfyUI_IPAdapter_plus
pip install -r requirements.txt
cd ..
```


Run ComfyUI

