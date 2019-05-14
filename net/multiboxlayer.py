
import torch
from torch import nn

from .module import SepConv2d


class MultiBoxLayer(nn.Module):
    def __init__(self, n_classes: int):
        super(MultiBoxLayer, self).__init__()

        self.n_classes = n_classes

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

    def forward(self, hs: list):
        '''
        Args:
            hs: (list) of (tensor)s that are intermediate layer outputs

        Returns:
            loc_preds: (tensor) predicted locations, shaped [n, _, 4]
            conf_preds: (tensor) predicted class confidences, shaped [n, _, n_classes]
        '''
        confidences = []
        locations = []

        for i, h in enumerate(hs):
            location = self.regression_heads[i](h)
            n = location.size(0)
            location = location.permute(0, 2, 3, 1).contiguous()
            location = location.view(n, -1, 4)
            locations.append(location)

            confidence = self.classification_heads[i](h)
            confidence = confidence.permute(0, 2, 3, 1).contiguous()
            confidence = confidence.view(n, -1, self.n_classes)
            confidences.append(confidence)

        loc_preds = torch.cat(locations, 1)
        conf_preds = torch.cat(confidences, 1)
        return loc_preds, conf_preds
