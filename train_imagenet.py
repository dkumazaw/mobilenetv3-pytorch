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

from mylogger import create_logger
from net.mobilenetv3 import MobileNetV3Large, MobileNetV3Small
from trainer import Trainer
import utils

BATCH_SIZE = 240
NUM_WORKERS = 16
EPOCHS = 1


def main():

    # Initialize the folder in which all training results will be saved
    model_save_dir = './models/imagenet-{}'.format(
        datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    logger = create_logger(
        filename=os.path.join(model_save_dir, 'log.log'),
        logger_prefix=__file__
    )

    model = MobileNetV3Large(n_classes=1000)
    if torch.cuda.device_count() > 1:
        logger.info('Parallelize by using {} available GPUs'.format(
            torch.cuda.device_count())
        )
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.1, alpha=0.9999, momentum=0.9, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.01
    )

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3,
                               contrast=0.3,
                               saturation=0.3,
                               hue=0.1),
        transforms.ToTensor(),
        utils.Cutout(20),
        normalizer
    ])

    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
        train_val_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True
    )

    valid_loader = DataLoader(
        train_val_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    epochs = EPOCHS

    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        epochs=epochs, logger=logger, model_save_dir=model_save_dir
    )

    trainer.train()
    trainer.validate()


if __name__ == '__main__':
    main()
