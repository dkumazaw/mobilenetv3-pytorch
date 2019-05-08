from torch import nn
from module import *


def _gen_init_conv_bn(in_dim: int, out_dim: int, stride: int):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_dim),
        HardSwish()
    )


def _gen_block_layer(config: list):
    op, kernel_size, hidden_dim, in_dim, out_dim, se, nl, stride = config
    return Block(in_dim, out_dim, hidden_dim, kernel_size, stride, nl, se)


def _gen_final_layer_bn(in_dim: int, hidden_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        HardSwish(),
        nn.AvgPool2d(7),
        HardSwish(),
        nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
        HardSwish()
    )


def _gen_final_layer_no_bn(in_dim: int, hidden_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        HardSwish(),
        nn.AvgPool2d(7),
        HardSwish(),
        nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
        nn.BatchNorm2d(out_dim),
        HardSwish()
    )


def _gen_classifier(in_dim: int, out_dim: int):
    return nn.Conv2d(in_dim, out_dim, 1, bias=False)


class MobileNetV3Large(nn.Module):
    def __init__(self, n_classes=1000, input_size=224):
        super(MobileNetV3Large, self).__init__()
        self._features = []

        # [op, kernel_size, hidden_dim(exp size), in_dim, out_dim(#out), SE, NL, s]
        self._layer_configs = [['bneck',  3,   16,  16,  16, False, 'RE', 1],
                               ['bneck',  3,   64,  16,  24, False, 'RE', 2],
                               ['bneck',  3,   72,  24,  24, False, 'RE', 1],
                               ['bneck',  5,   72,  24,  40,  True, 'RE', 2],
                               ['bneck',  5,  120,  40,  40,  True, 'RE', 1],
                               ['bneck',  5,  120,  40,  40,  True, 'RE', 1],
                               ['bneck',  3,  240,  40,  80, False, 'HS', 2],
                               ['bneck',  3,  200,  80,  80, False, 'HS', 1],
                               ['bneck',  3,  184,  80,  80, False, 'HS', 1],
                               ['bneck',  3,  184,  80,  80, False, 'HS', 1],
                               ['bneck',  3,  480,  80, 112,  True, 'HS', 1],
                               ['bneck',  3,  672, 112, 112,  True, 'HS', 1],
                               ['bneck',  5,  672, 112, 160,  True, 'HS', 1],
                               ['bneck',  5,  672, 160, 160,  True, 'HS', 2],
                               ['bneck',  5,  960, 160, 160,  True, 'HS', 1]]

        # First layer
        self._features.append(_gen_init_conv_bn(3, 16, 2))

        for config in self._layer_configs:
            self._features.append(_gen_block_layer(config))

        # Final layer
        self._features.append(_gen_final_layer_bn(160, 960, 1280))

        self._features = nn.Sequential(*self._features)

        # Classifier
        self._classifier = _gen_classifier(1280, n_classes)

    def forward(self, x):
        x = self._features(x)
        x = self._classifier(x)
        return x


class MobileNetV3Small(nn.Module):
    def __init__(self, n_classes=1000, input_size=224):
        super(MobileNetV3Small, self).__init__()
        self._features = []

        # [op, kernel_size, hidden_dim(exp size), in_dim, out_dim(#out), SE, NL, s]
        self._layer_configs = [['bneck',  3,   16,  16,  16,  True, 'RE', 2],
                               ['bneck',  3,   16,  72,  24, False, 'RE', 2],
                               ['bneck',  3,   24,  88,  24, False, 'RE', 1],
                               ['bneck',  5,   24,  96,  40,  True, 'HS', 1],
                               ['bneck',  5,   40, 240,  40,  True, 'HS', 1],
                               ['bneck',  5,   40, 240,  40,  True, 'HS', 1],
                               ['bneck',  5,   40, 240,  40,  True, 'HS', 1],
                               ['bneck',  5,   40, 120,  48,  True, 'HS', 1],
                               ['bneck',  5,   48, 144,  48,  True, 'HS', 1],
                               ['bneck',  5,   48, 288,  96,  True, 'HS', 2],
                               ['bneck',  5,   96, 576,  96,  True, 'HS', 1],
                               ['bneck',  5,   96, 576,  96,  True, 'HS', 1]]
        # First layer
        self._features.append(_gen_init_conv_bn(3, 16, 2))

        for config in self._layer_configs:
            self._features.append(_gen_block_layer(config))

        # Final layer
        self._features.append(_gen_final_layer_no_bn(96, 576, 1280))

        self._features = nn.Sequential(*self._features)

        # Classifier
        self._classifier = _gen_classifier(1280, n_classes)

    def forward(self, x):
        x = self._features(x)
        x = self._classifier(x)
        return x
