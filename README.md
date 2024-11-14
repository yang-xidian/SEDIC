# Taming Large Multimodal Agents for Ultra-low bitrate Semantically Distangled Image Compression


[![PDF Thumbnail](https://github.com/yang-xidian/SEDIC/blob/main/method.jpg)](https://github.com/yang-xidian/SEDIC/blob/main/method.pdf)

# Result
[![PDF Thumbnail](https://github.com/yang-xidian/SEDIC/blob/main/vision_image.jpg)](https://github.com/yang-xidian/SEDIC/blob/main/vision_image.pdf)

# Dependency
[GPT-4 Vision](https://openai.com/)

[Stable Diffusion 2.1](https://hf-mirror.com/stabilityai/stable-diffusion-2-1)

[DiffBIR](https://github.com/XPixelGroup/DiffBIR/)

[CompressAI](https://github.com/InterDigitalInc/CompressAI)

[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

[SAM](https://segment-anything.com/)

# Preparation

## Installation

A suitable conda environment named SEDIC can be created and activated with:

```bash
conda env create -n CL-LRPE python=3.8.1
conda activate CL-LRPE 
```

Install environment
```bash
pip install -r requirements.txt
```

## Weights
Download weights and put them into the weight folder:

DiffBIR (general_full_v1.ckpt): [link](https://hf-mirror.com/lxq007/DiffBIR-v2/resolve/main/v1_general.pth)

Cheng2020-Tuned (cheng_small.pth.tar): [link](https://www.dropbox.com/scl/fi/br0zu6a91wygs68afesyo/cheng_small.pth.tar?rlkey=2gdhpy3z5qank0giajacj2u9p&st=9q2y88aj&dl=0)

GroundingDINO (GroundingDINO-T): [link](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

SAM (sam_vit_h_4b8939.pth): [link](https://github.com/facebookresearch/segment-anything#model-checkpoints)
