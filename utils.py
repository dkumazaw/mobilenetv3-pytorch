import os
import shutil

import torch


class AveTracker:
    def __init__(self):
        self.average = 0
        self.sum = 0
        self.counter = 0

    def update(self, value, n):
        self.sum += value * n
        self.counter += n
        self.average = self.sum / self.counter


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state_dict, is_best, savedir, epoch):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filename = os.path.join(savedir, 'checkpoint_ep{}.pt'.format(epoch))
    torch.save(state_dict, filename)

    if is_best:
        best_filename = os.path.join(savepath, 'model_best.pt')
        shutil.copyfile(filename, best_filename)
