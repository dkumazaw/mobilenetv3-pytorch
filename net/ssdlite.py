import torch
from torch import nn

from .module import Block, SepConv2d
from .multiboxlayer import MultiBoxLayer


# Inspired by https://github.com/amdegroot/ssd.pytorch/ and https://github.com/qfgaohao/pytorch-ssd/
class SSDLite(nn.Module):
    """SSD Lite

    Uses a MobileNetV3 passed as base_net as the backbone network.
    """

    def __init__(self, base_net, n_classes, heads):
        super(SSDLite, self).__init__()

        self._base_layers = nn.ModuleList()

        self.extras = nn.ModuleList([
            Block(in_channels=960, out_channels=512,
                  hidden_channels=int(1280 * 0.2)),
            Block(in_channels=512, out_channels=256,
                  hidden_channels=int(512 * 0.25)),
            Block(in_channels=256, out_channels=256,
                  hidden_channels=int(256 * 0.5)),
            Block(in_channels=256, out_channels=64,
                  hidden_channels=int(256 * 0.25))
        ])
        self.regression_heads = nn.ModuleList([
            SepConv2d(in_channels=672, out_channels=6 * 4),
            SepConv2d(in_channels=960, out_channels=6 * 4),
            SepConv2d(in_channels=512, out_channels=6 * 4),
            SepConv2d(in_channels=256, out_channels=6 * 4),
            SepConv2d(in_channels=256, out_channels=6 * 4),
            nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ])
        self.classification_heads = nn.ModuleList([
            SepConv2d(in_channels=672, out_channels=6 * n_classes),
            SepConv2d(in_channels=960, out_channels=6 * n_classes),
            SepConv2d(in_channels=512, out_channels=6 * n_classes),
            SepConv2d(in_channels=256, out_channels=6 * n_classes),
            SepConv2d(in_channels=256, out_channels=6 * n_classes),
            nn.Conv2d(in_channels=64, out_channels=6 *
                      n_classes, kernel_size=1)
        ])

        self._multibox_layer = MultiBoxLayer(
            n_classes=n_classes,
            regression_heads=self.regression_heads,
            classification_heads=self.classification_heads
        )

    def forward(self, x):
        confs = []
        locs = []
        pass
