import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.autoencoder import BaseAutoencoder


class CNNAutoencoder(BaseAutoencoder):

    def __init__(self, hparams, example_input):
        super().__init__(hparams)
        self.encoder = torch.nn.Sequential(
            torch.nn.Upsample((64, 64), mode='bilinear'),
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, 3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 3),
            torch.nn.Upsample(example_input.shape, mode='bilinear')
        )
