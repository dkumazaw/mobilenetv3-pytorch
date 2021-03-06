# MobileNetV3 PyTorch implementation
This is an unofficial implementation of [MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) in PyTorch. Currently this repo contains the small and large versions of MobileNetV3, but I plan to also implement detection and segmentation extensions.

## How this repo is organized
- `net/` folder contains model definitions
- `data/` folder is used when running training code
- `models/` will store checkpoints & training log
- `train_*.py` runs training

## How to use the models:
Models are found under `net` folder. You can load models as follows:
```python
from net.mobilenetv3 import MobileNetV3Large, MobileNetV3Small

model_large = MobileNetV3Large(n_classes=1000) # Or use small
```

## Train on CIFAR10
The script `train_cifar10.py` pulls the CIFAR10 dataset using [torchvision datasets](https://pytorch.org/docs/stable/torchvision/datasets.html) and trains a MobileNetV3 on it. Note that the dimension was upsampled to 224x224 in order to match the dimensions.

You can start the training by simply executing:
```
python train_cifar10.py
```

### Performance 
(WIP)

## Train on ImageNet
To train the models on ImageNet, run `train_imagenet.py`. The script assumes that the imagenet dataset is placed under `data/imagenet/` folder. 

### Performance
(WIP)

## SSDLite (WIP)
Implemented in `net/ssdlite.py`

## TODO:
- [ ] Training code for ImageNet
- [ ] Detection: SSDLite
- [ ] Segmentation:  Lite R-ASPP
