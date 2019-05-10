# Inspired by https://github.com/victoresque/pytorch-template/
import logging
import sys

import torch

import utils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer:
    def __init__(self, model, criterion, optimizer,
                 device, train_loader, valid_loader, epochs, model_save_dir='./models/cifar'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_save_dir = model_save_dir
        self.epochs = epochs

        self.model = self.model.to(device)

    def train(self):
        """Trains the model for epochs"""

        for epoch in range(self.epochs):
            logging.info('epoch %d', epoch)
            best_valid_loss = float('inf')

            # Training
            train_top1_acc, train_top5_acc, train_loss = self._train_epoch(
                epoch)
            logging.info('train_top1_acc {:.5f}, train_top5_acc {:.5f}, train_loss {:.5f}'.format(
                train_top1_acc, train_top5_acc, train_loss))

            # Validation
            valid_top1_acc, valid_top5_acc, valid_loss = self._valid_epoch(
                epoch)
            logging.info('valid_top1_acc {:.5f}, valid_top5_acc {:.5f}, valid_loss {:.5f}'.format(
                valid_top1_acc, valid_top5_acc, valid_loss))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                is_best = True
            else:
                is_best = False

            utils.save_checkpoint(self.model.state_dict(),
                                  is_best, self.model_save_dir, epoch)

    def _train_epoch(self, epoch: int):
        """Trains the model for one epoch"""
        total_loss = utils.AveTracker()
        top1_acc = utils.AveTracker()
        top5_acc = utils.AveTracker()
        self.model.train()

        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            n = x.size(0)
            total_loss.update(loss.item(), n)
            top1_acc.update(prec1.item(), n)
            top5_acc.update(prec5.item(), n)

            if step % 100 == 0:
                logging.info('train %d %e %f %f', step,
                             total_loss.average, top1_acc.average, top5_acc.average)

        return top1_acc.average, top5_acc.average, total_loss.average

    def _valid_epoch(self, epoch):
        """Runs validation"""
        total_loss = utils.AveTracker()
        top1_acc = utils.AveTracker()
        top5_acc = utils.AveTracker()
        self.model.eval()

        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
                n = x.size(0)
                total_loss.update(loss.item(), n)
                top1_acc.update(prec1.item(), n)
                top5_acc.update(prec5.item(), n)

                if step % 100 == 0:
                    logging.info('valid %d %e %f %f', step,
                                 total_loss.average, top1_acc.average, top5_acc.average)

        return top1_acc.average, top5_acc.average, total_loss.average
