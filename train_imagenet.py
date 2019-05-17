from datetime import datetime
import logging
import os
import random

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms

from mylogger import create_logger
from net.mobilenetv3 import MobileNetV3Large, MobileNetV3Small
from trainer import ClassifierTrainer as Trainer
import utils

BATCH_SIZE = 245
NUM_WORKERS = 16
EPOCHS = 20


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

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5
    )

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
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

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalizer
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        './data/imagenet/train', transform=transform_train
    )

    valid_dataset = torchvision.datasets.ImageFolder(
        './data/imagenet/valid', transform=transform_test
    )

    test_dataset = torchvision.datasets.ImageFolder(
        './data/imagenet/test', transform=transform_test
    )

    scheduler = utils.OneCycleLR(
        optimizer,
        num_steps=int((len(train_dataset)/BATCH_SIZE) * EPOCHS),
        lr_range=(0.2, 0.8),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
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
