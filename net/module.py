from torch import nn


class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()
        self._relu6_layer = nn.ReLU6()

    def forward(self, x):
        return self._relu6_layer(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self._hard_sigmoid = HardSigmoid()

    def forward(self, x):
        return x * self._hard_sigmoid(x)


# Refer to https://github.com/moskomule/senet.pytorch/
class SqueezeAndExcite(nn.Module):
    def __init__(self, channel):
        super(SqueezeAndExcite, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            HardSigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self._avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, kernel_size: int, stride: int, nl: str, se: bool):
        super(Block, self).__init__()

        if nl == 'HS':
            self._non_linearity = HardSwish
        elif nl == 'RE':
            self._non_linearity = nn.ReLU6
        else:
            raise ValueError('Non-linearity must be either HS or RE.')

        if se:
            self._layers = nn.Sequential(
                # 1x1 w/o activation
                nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # kernel_size x kernel_size depthwise w/ activation
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          1, groups=hidden_dim, bias=False),
                SqueezeAndExcite(hidden_dim),  # Squeeze and excite
                nn.BatchNorm2d(hidden_dim),
                self._non_linearity(),
                # 1x1 w/ activation
                nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_dim),
                self._non_linearity()
            )
        else:
            self._layers = nn.Sequential(
                # 1x1 w/o activation
                nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # kernel_size x kernel_size depthwise w/ activation
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                self._non_linearity(),
                # 1x1 w/ activation
                nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_dim),
                self._non_linearity()
            )

    def forward(self, x):
        return self._layers(x)
