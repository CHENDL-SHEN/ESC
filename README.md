

# SP-CAM


## Abstract
Weakly supervised semantic segmentation (WSSS) using only image-level labels can significantly reduce the annotation cost and thus has attracted considerable research attention in recent years. Most advanced WSSS methods use the class activation maps (CAMs) to generate pseudo labels for training a segmentation network.
However, the low-resolution CAMs often lead to low-quality pseudo labels. To overcome this limitation, this paper proposes a novel WSSS framework, where a superpixel-level class activation map (SP-CAM) is introduced to obtain explicit pseudo labels. 
First, the fully convolutional  network with an improved loss is used to generate the pixel-superpixel association map (PSAM) and the adjacent superpixel affinity matrix (ASAM). Then, a deconvolution reconstruction module  is developed to approximate SP-CAMs. Finally, the SP-CAMs are revised by the proposed post-processing schemes combined with the superpixel-based random walk and PSAM upsampling, thus obtaining more explicit pseudo labels. 
Extensive experiments demonstrate the high effectiveness of the proposed method, which achieves the new state-of-the-art performance on both PASCAL VOC 2012 and MS COCO 2014 datasets.
## Overview
![Overall architecture](./figures/process.png)

<br>

## Prerequisite
- Python 3.6, PyTorch 1.8.0, and more in requirements.txt
- CUDA 11.1
- 4 x  RTX 3090 GPUs

## Usage

### Install python dependencies
```bash
python3 -m pip install -r requirements.txt
```
### Dataset Download
- PASCAL VOC 2012
    - [Images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 
    - [Saliency maps](https://drive.google.com/file/d/1Za0qNuIwG64-eteuz5SMbWVFL6udsLWd/view?usp=sharing) 
      using [PoolNet]

- MS-COCO 2014
    - [Images](https://cocodataset.org/#home) 
    - [Saliency maps](https://drive.google.com/file/d/1amJWDeLOj567JQMGGsSyqi7-g65dWxr0/view?usp=sharing)  using [PoolNet] 
    - [Segmentation masks](https://drive.google.com/file/d/16wuPinx0rdIP_PO0uYeCn9rfX2-evc-S/view?usp=sharing)


### Pseudo Labels Generation network  
- Set the datasets root at ```dataset_root.py```.
    ```python
    # \dataset_root.py.
    VOC_ROOT='VOC2012/VOCdevkit/VOC2012/'
    VOC_SAL_ROOT='VOC2012/VOCdevkit/VOC2012/saliency_map/'
    COCO_ROOT='COCO/'
    COCO_SAL_ROOT = 'COCO/saliency_maps_poolnet/'
- Execute the bash file for the three-stage training process.
    ```bash
    # Please see these files for the detail of execution.

    bash scripts/train_all.sh
- Also, the trained FCN model parameters is provided in this repo, you can skip the first two steps and execute the third step directly to generate SP-CAMs.
    ```bash
    # Please see these files for the detail of execution.

    bash script/train_sp_cam.sh
### Segmentation network
- We utilize [DeepLab-V2](https://arxiv.org/abs/1606.00915) 
  for the segmentation network. 
- Please see [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) for the implementation in PyTorch.
  
## Resluts
### trained  checkpoints of SP-CAM mdoel
#### voc
[best model (mIou 79.10%)](https://drive.google.com/file/d/1mYTvFK-W7le_5Q-vdeiyyHZMkEYj1Q8H/view?usp=sharing)
#### coco
 [ model (mIou 48.37%)](https://drive.google.com/file/d/1YR_fxL8TNILKjE6gDK7a8wHIt--8g1qP/view?usp=sharing)
<!-- ### segmentation mdoel -->
