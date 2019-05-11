from torch import nn


# Inspired by implementation available from https://github.com/amdegroot/ssd.pytorch/
class SSDLite(nn.Module):
    """SSD Lite
    
    Uses a MobileNetV3 passed as base_net as the backbone network.
    """
    def __init__(self, base_net, n_classes, heads):
        super(SSDLite, self).__init__()

        self.classification_heads = 
        self.regression_heads = regression_heads

    
    def forward(self, x):
        pass