import torch
from torch import nn

from .module import *
from .mobilenetv3 import *
from .multiboxlayer import MultiBoxLayer

# Inspired by https://github.com/amdegroot/ssd.pytorch/ and https://github.com/qfgaohao/pytorch-ssd/


class SSDLite(nn.Module):
    """SSD Lite

    Uses a MobileNetV3 passed as base_net as the backbone network.
    """

    def __init__(self, n_classes: int):
        super(SSDLite, self).__init__()

        self._n_classes = n_classes
        self._base_layers = nn.ModuleList()

        # [kernel_size, hidden_channels(exp size), in_channels, out_channels(#out), SE, NL, s]
        self._block_layer_configs = [[3,   16,  16,  16, False, 'RE', 1, False],
                                     [3,   64,  16,  24, False, 'RE', 2, False],
                                     [3,   72,  24,  24, False, 'RE', 1, False],
                                     [5,   72,  24,  40,  True, 'RE', 2, False],
                                     [5,  120,  40,  40,  True, 'RE', 1, False],
                                     [5,  120,  40,  40,  True, 'RE', 1, False],
                                     [3,  240,  40,  80, False, 'HS', 2, False],
                                     [3,  200,  80,  80, False, 'HS', 1, False],
                                     [3,  184,  80,  80, False, 'HS', 1, False],
                                     [3,  184,  80,  80, False, 'HS', 1, False],
                                     [3,  480,  80, 112,  True, 'HS', 1, False],
                                     [3,  672, 112, 112,  True, 'HS', 1, False],
                                     [5,  672, 112, 160,  True, 'HS', 1, True],
                                     [5,  672, 160, 160,  True, 'HS', 2, False],
                                     [5,  960, 160, 160,  True, 'HS', 1, False]]

        self._base_layers.append(gen_init_conv_bn(3, 16, 2))

        for config in self._block_layer_configs:
            self._base_layers.append(gen_block_layer(config))

        self._base_layers.append(nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish(),
        ))

        self._extra_layers = nn.ModuleList([
            Block(in_channels=960, out_channels=512,
                  hidden_channels=int(1280 * 0.2)),
            Block(in_channels=512, out_channels=256,
                  hidden_channels=int(512 * 0.25)),
            Block(in_channels=256, out_channels=256,
                  hidden_channels=int(256 * 0.5)),
            Block(in_channels=256, out_channels=64,
                  hidden_channels=int(256 * 0.25))
        ])

        self._multibox_layer = MultiBoxLayer(n_classes=self._n_classes)

    def forward(self, x):
        confs = []
        locs = []
        hs = []
        # Forward pass through base layers
        for index, layer in enumerate(self._base_layers[:-1]):
            if index == 13:
                # C4 layer
                x, h = layer(x)
                hs.append(h)
                continue

            x = layer(x)

        # Last layer in base layers
        x = self._base_layers[-1](x)
        hs.append(x)

        # Forward pass through extra layers
        for layer in self._extra_layers:
            x = layer(x)
            hs.append(x)

        loc_preds, conf_preds = self._multibox_layer(hs)
        return loc_preds, conf_preds
