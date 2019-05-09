# MobileNetV3 unofficial PyTorch implementation
This is an unofficial implementation of [MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) in PyTorch. Currently this repo contains the small and large versions of MobileNetV3, but I plan to also implement detection and segmentation extensions.

## How to use the models:
Models are found under `net` folder. You can load models as follows:
```python
from net.mobilenetv3 import MobileNetV3Large, MobileNetV3Small

model_large = MobileNetV3Large(n_classes=1000) # Or use small
```

## Train the model on CIFAR10 [WIP]
Simply run
```
python train_cifar10.py
```

## TODO:
- [ ] training code for ImageNet
- [ ] Detection: SSDLite
- [ ] Segmentation:  Lite R-ASPP
