# comfyui_bmab

BMAB is an custom nodes of ComfyUI and has the function of post-processing the generated image according to settings.
If necessary, you can find and redraw people, faces, and hands, or perform functions such as resize, resample, and add noise.
You can composite two images or perform the Upscale function.

<img src="https://i.ibb.co/341r93k/2024-05-21-10-56-02.png"/>

You can download sample.json.

https://github.com/portu-sim/comfyui_bmab/blob/main/resources/sample/sample.json


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

## Install Manually

I can't describe about your python environment.
I will write the installation instructions assuming you have some knowledge of Python.

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

### Windows & Linux

visit https://github.com/Bing-su/GroundingDINO/releases
Copy link for suitable package URL.

Windows, Pytorch 2.2, Cuda 12.1, Python 3.10

```commandline
pip install https://github.com/Bing-su/GroundingDINO/releases/download/v24.5.19/groundingdino-24.5.19+torch2.2.2.cu121-cp310-cp310-win_amd64.whl
```

### MacOS or Unknown

The installation method below is most likely to work with the CPU. So it can be very slow.

```commandline
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```


Run ComfyUI

