

# SP-CAM


## Abstract
Weakly supervised semantic segmentation (WSSS) using
only image-level labels can significantly reduce annotation
costs and attract considerable research attention. Most
advanced WSSS methods exploit the class activation maps
(CAMs) to generate pseudo labels for training the segmentation network. However, the low-resolution CAMs often
lead to low-quality pseudo labels. To overcome this challenge, we propose a novel WSSS framework, in which the
superpixel level class activation map (SP-CAM) is introduced to obtain the explicit pseudo labels. Firstly, the
FCN network with the improved loss is used to generate the
pixel-superpixel association map (PSAM) and the adjacent
superpixel affinity matrix (ASAM). Secondly, a deconvolution reconstruction module (DRM) is devised to approximate SP-CAMs. Finally, SP-CAMs are revised by our postprocessing schemes combining with the ASAM diffusion and
PSAM upsampling, leading to more explicit pseudo labels.
Experimental results on PASCAL VOC 2012 dataset demonstrate the effectiveness of the proposed method, which
yields the new record of the mIoU metric in the weaklysupervised semantic segmentation. 

## Overview
![Overall architecture](./figures/process.png)

<br>

# Prerequisite
- Python 3.6, PyTorch 1.8.0, and more in requirements.txt
- CUDA 11.1, cuDNN 7.6.5
- 4 x  RTX 3090 GPUs

# Usage

## Install python dependencies
```bash
python3 -m pip install -r requirements.txt
```
### Dataset & pretrained model
- PASCAL VOC 2012
    - [Images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 
    - [Saliency maps]() 
      using [PoolNet](https://arxiv.org/pdf/1904.09569v1.pdf)

- MS-COCO 2014
    - [Images](https://cocodataset.org/#home) 
    - [Saliency maps]() 
      using [PoolNet](https://arxiv.org/pdf/1904.09569v1.pdf)
    - [Segmentation masks](https://drive.google.com/file/d/16wuPinx0rdIP_PO0uYeCn9rfX2-evc-S/view?usp=sharing)



