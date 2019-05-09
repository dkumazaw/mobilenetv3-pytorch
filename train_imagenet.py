import torch
from torch import nn

import torchvision
from torchvision import transforms

from net.mobilenetv3 import MobileNetV3Large, MobileNetV3Small


def main():

    model = MobileNetV3Large()
    criterion = nn.CrossEntropyLoss()

    optimizer = nn.optim.Adam(lr=3e-4)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    train_loader = None  # TODO: Implement this
    valid_loder = None

    trainer = Traner(model=model, criterion=criterion, optimizer=optimizer,
                     device=device, train_loader=train_loader, valid_loader=valid_loder)

    epochs = 50
    trainer.train(epochs=epochs)


if __name__ == '__main__':
    main()
