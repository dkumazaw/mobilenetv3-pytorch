from torch import nn


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self._relu6_layer = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self._relu6_layer(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self._hard_sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self._hard_sigmoid(x)


class SqueezeAndExcite(nn.Module):
    def __init__(self, channel, reduce_factor=4):
        super(SqueezeAndExcite, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduce_factor),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduce_factor, channel),
            HardSigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self._avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y


class SepConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SepConv2d, self).__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self._layers(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 kernel_size: int = 3, stride: int = 2,
                 nl: str = 'RE', se: bool = False, return_intermed: bool = False):
        '''
        Args:
            in_channels:     (int)  # of channels of input tensor
            out_channels:    (int)  # of channels of output tensor
            hidden_channels: (int)  # of channels of intermediate tensor after expansion layer
            kernel_size:     (int)  kernel size in conv operation
            stride:          (int)  stride in conv operation
            nl:              (str)  non linearity, either 'RE' for ReLU or 'HS' for hard swish
            se:              (bool) True if apply squeeze and excitation
            return_intermed  (bool) True if want to return the intermediate tensor of the expansion layer for SSDLite
        '''
        super(Block, self).__init__()

        if nl == 'HS':
            self._non_linearity = HardSwish
        elif nl == 'RE':
            self._non_linearity = nn.ReLU6
        else:
            raise ValueError('Non-linearity must be either HS or RE.')

        self._return_intermed = return_intermed

        self._will_skipconnect = stride == 1 and in_channels == out_channels

        self._expand = nn.Sequential(
            # expansion layer: 1x1 w/o activation
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )

        conv_and_reduce_layers = [
            # kernel_size x kernel_size depthwise w/ activation
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size//2, groups=hidden_channels, bias=False),
        ]

        if se:
            conv_and_reduce_layers.append(
                SqueezeAndExcite(hidden_channels)  # Squeeze and excite
            )

        conv_and_reduce_layers.extend([
            nn.BatchNorm2d(hidden_channels),
            self._non_linearity(),

            # 1x1 w/ activation
            nn.Conv2d(hidden_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self._non_linearity()
        ])

        self._conv_and_reduce = nn.Sequential(*conv_and_reduce_layers)

    def forward(self, x):
        h = self._expand(x)
        out = self._conv_and_reduce(h)
        if self._will_skipconnect:
            out = out + x

        if self._return_intermed:
            return out, h
        else:
            return out
