from torch import nn

from .module import Block, SepConv2d


# Inspired by implementation available from https://github.com/amdegroot/ssd.pytorch/
class SSDLite(nn.Module):
    """SSD Lite
    
    Uses a MobileNetV3 passed as base_net as the backbone network.
    """
    def __init__(self, base_net, n_classes, heads):
        super(SSDLite, self).__init__()

        self.extras = nn.ModuleList([
            Block(in_channels=1280, out_channels=512, hidden_channels=int(1280 * 0.2)),
            Block(in_channels=512,  out_channels=256, hidden_channels=int(512 * 0.25)),
            Block(in_channels=256,  out_channels=256, hidden_channels=int(256 * 0.5)),
            Block(in_channels=256,  out_channels=64,  hidden_channels=int(256 * 0.25))
        ])
        self.regression_headers = nn.ModuleList([
            SepConv2d(in_channels)
            SepConv2d(in_channels=1280, out_channels=6 * 4),
            SepConv2d(in_channels=512, out_channels=6 * 4),
            SepConv2d(in_channels=256, out_channels=6 * 4),
            SepConv2d(in_channels=256, out_channels=6 * 4),
            Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ])
        self.regression_heads = nn.ModuleList([
            SepConv2d(in_channels=)
        ])

    
    def forward(self, x):
        pass