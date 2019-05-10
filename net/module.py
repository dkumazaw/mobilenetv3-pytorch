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


class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, kernel_size: int, stride: int, nl: str, se: bool):
        super(Block, self).__init__()

        if nl == 'HS':
            self._non_linearity = HardSwish
        elif nl == 'RE':
            self._non_linearity = nn.ReLU6
        else:
            raise ValueError('Non-linearity must be either HS or RE.')

        self._will_skipconnect = stride == 1 and in_dim == out_dim

        if se:
            self._layers = nn.Sequential(
                # 1x1 w/o activation
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # kernel_size x kernel_size depthwise w/ activation
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size//2, groups=hidden_dim, bias=False),
                SqueezeAndExcite(hidden_dim),  # Squeeze and excite
                nn.BatchNorm2d(hidden_dim),
                self._non_linearity(),
                # 1x1 w/ activation
                nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                self._non_linearity()
            )
        else:
            self._layers = nn.Sequential(
                # 1x1 w/o activation
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # kernel_size x kernel_size depthwise w/ activation
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                self._non_linearity(),
                # 1x1 w/ activation
                nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                self._non_linearity()
            )

    def forward(self, x):
        out = self._layers(x)
        if self._will_skipconnect:
            return out + x
        else:
            return out
