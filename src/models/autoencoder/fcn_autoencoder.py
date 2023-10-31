import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.modules.activation import ReLU

from src.models.autoencoder import BaseAutoencoder


class FCNAutoencoder(BaseAutoencoder):

    def __init__(self, hparams, example_input):
        super().__init__(hparams)
        hidden_dim = 32
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(example_input.size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, example_input.size),
        )
