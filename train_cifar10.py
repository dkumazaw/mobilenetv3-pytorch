import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms

from net.mobilenetv3 import MobileNetV3Large, MobileNetV3Small
from trainer import Trainer
import utils


def main():

    model = MobileNetV3Large(n_classes=10)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),  # Upsample
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((224, 224)),  # Upsample
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        './data/cifar/train', train=True, transform=transform_train, download=True
    )

    valid_test_dataset = torchvision.datasets.CIFAR10(
        './data/cifar/test', train=False, transform=transform_valid, download=True
    )

    # Split valid_test_dataset into two
    dataset_size = len(valid_test_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    valid_indices = indices[:dataset_size//2]
    test_sampler = indices[dataset_size//2:]

    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_sampler)

    train_loader = DataLoader(
        train_dataset, batch_size=200, shuffle=True, num_workers=8
    )

    valid_loder = DataLoader(
        valid_test_dataset, batch_size=200, sampler=valid_sampler, num_workers=8
    )

    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        device=device, train_loader=train_loader, valid_loader=valid_loder
    )

    epochs = 1
    trainer.train(epochs=epochs)


if __name__ == '__main__':
    main()
