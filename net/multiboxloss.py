from torch import nn


class MultiBoxLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, loc_preds, conf_preds, gt_locs, gt_labels):
