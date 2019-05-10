from datetime import datetime
import logging
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms

from net.mobilenetv3 import MobileNetV3Large, MobileNetV3Small
import utils


def main():

    model = MobileNetV3Large(n_classes=1000)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.1, alpha=0.9999, momentum=0.9, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.01
    )

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(3),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3,
                               contrast=0.3,
                               saturation=0.3,
                               hue=0.1),
        transforms.ToTensor(),
        utils.Cutout(20),
        normalizer
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])

    train_val_dataset = torchvision.datasets.ImageFolder(
        './data/imagenet/train', transform=transform_train
    )

    test_dataset = torchvision.datasets.ImageFolder(
        './data/imagenet/valid', transform=transform_valid
    )

    # Create validation dataset
    dataset_size = len(train_val_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_indices = indices[50000:]
    valid_indices = indices[:50000]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(
        train_val_dataset, batch_size=220, sampler=train_sampler, num_workers=8
    )

    valid_loader = DataLoader(
        train_val_dataset, batch_size=220, sampler=valid_sampler, num_workers=8
    )

    test_loader = DataLoader(
        test_dataset, batch_size=220, num_workers=8
    )

    model_save_dir = './models/imagenet-{}'.format(
        datetime.now().strftime('%Y%m%d-%H%M%S'))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    log_name = os.path.join(model_save_dir, 'log.log')

    logging.basicConfig(filename=log_name,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger(__file__)

    epochs = 50

    trainer = Traner(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, train_loader=train_loader, valid_loader=valid_loder,
        epochs=epochs, logger=logger, model_save_dir=model_save_dir
    )

    trainer.train()


if __name__ == '__main__':
    main()
