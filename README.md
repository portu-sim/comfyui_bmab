# comfyui_bmab

BMAB is an custom nodes of ComfyUI and has the function of post-processing the generated image according to settings.
If necessary, you can find and redraw people, faces, and hands, or perform functions such as resize, resample, and add noise.
You can composite two images or perform the Upscale function.

<img src="https://i.ibb.co/341r93k/2024-05-21-10-56-02.png"/>

You can download sample.json.

https://github.com/portu-sim/comfyui_bmab/blob/main/resources/examples/example.json


### Hand Detailing Sample

BMAB detects and enlarges the upper body of a person and performs Openpose at high resolution to fix incorrectly drawn hands.

<img src="https://i.ibb.co/LRf0DKv/2024-05-23-1-42-12.png"/>


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

And be sure to install <span style="color:red"> GroundingDINO </span>  at the bottom.


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


## Install GroundingDINO

comfyui_bmab requires GroundingDINO for some detection processing.   
However, this cannot be installed using pip, so follow the following procedure.

### Windows - CompyUI Portable User

```commandline
python_embeded\python.exe -m pip install https://github.com/portu-sim/GroundingDINO/releases/download/groundingdino-0.1.0/groundingdino-0.1.0.torch2.3.0.cu121-cp311-cp311-win_amd64.whl
```

### Windows

```commandline
pip3 install https://github.com/portu-sim/GroundingDINO/releases/download/groundingdino-0.1.0/groundingdino-0.1.0.torch2.3.0.cu121-cp311-cp311-win_amd64.whl
```

### Linux

visit https://github.com/Bing-su/GroundingDINO/releases
Copy link for suitable package URL.

You can check versions in python

```commandline
>>>
>>> import sys
>>> sys.version
'3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)]'
>>> import torch
>>> torch.__version__
'2.1.2+cu121'
>>>
```


Windows, Pytorch 2.2, Cuda 12.1, Python 3.10

```commandline
pip install https://github.com/Bing-su/GroundingDINO/releases/download/v24.5.19/groundingdino-24.5.19+torch2.2.2.cu121-cp310-cp310-win_amd64.whl
```

Windows, Pytorch 2.3, Cuda 12.1, Python 3.10

```commandline
pip install https://github.com/Bing-su/GroundingDINO/releases/download/v24.5.19/groundingdino-24.5.19+torch2.3.0.cu121-cp310-cp310-win_amd64.whl
```

### MacOS or Unknown

The installation method below is most likely to work with the CPU. So it can be very slow.

```commandline
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```


Run ComfyUI

