from turtle import forward
import numpy as np
import torch

import wandb

from src.models.forecasting import BaseForecasting


class Forecasting(BaseForecasting):
    """
    Deep Neural Network for processing energy and calendar data,
    where the energy data is processed by fully-connected layers.
    """

    def __init__(self, hparams, dataset):
        super().__init__(hparams, dataset)

    def _build_energy_net(self, example_energy, latent_energy_size=64):
        # energy processing part
        if self.hparams.energy_architecture.lower() == 'fcn':
            self._energy_net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(self.hparams.energy_lag_features, 2 * latent_energy_size),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm1d(2 * latent_energy_size),
                torch.nn.Linear(2 * latent_energy_size, latent_energy_size),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm1d(latent_energy_size)
            )
        elif self.hparams.energy_architecture.lower() == 'cnn':
            example_energy = example_energy.view(example_energy.shape[0], 1, -1)
            block = lambda input_size, features: [
                torch.nn.Conv1d(input_size, features, 3, padding='same'),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm1d(features),
                torch.nn.MaxPool1d(2),
            ]
            energy_cnn = torch.nn.Sequential(
                *block(1, 8),  # 168 -> 84
                *block(8, 4),  # 84 -> 42
            )
            energy_features = energy_cnn(example_energy)
            self._energy_net = torch.nn.Sequential(
                energy_cnn,
                torch.nn.Flatten(),
                torch.nn.Linear(np.prod(energy_features.shape[1:]), latent_energy_size),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm1d(latent_energy_size)
            )
        elif self.hparams.energy_architecture.lower() == 'flatten':
            self._energy_net = torch.nn.Flatten()
        else:
            raise NotImplementedError(f'Unkown energy architecture {self.hparams.energy_architecture} not implemented.')

    @torch.no_grad()
    def _build_weather_net(self, example_weather, latent_weather_size=128):
        # weather processing part
        if self.hparams.weather_architecture.lower() == 'fcn':
            self._weather_net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(np.prod(example_weather.shape[1:]), latent_weather_size),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm1d(latent_weather_size)
            )
        elif self.hparams.weather_architecture.lower() == '2dcnn':
            block = lambda input_size, features, groups: [
                torch.nn.Conv2d(input_size, features, 3,
                                groups=groups, padding='same'),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            ]
            groups = example_weather.shape[1]
            weather_cnn = torch.nn.Sequential(
                # torch.nn.Upsample(size=(32, 32), mode='bilinear'),
                *block(groups, 2 * groups, groups),  # 32 -> 14
                *block(2 * groups, 4 * groups, groups)  # 14 -> 5
            )
            weather_cnn_features = weather_cnn(example_weather)
            self._weather_net = torch.nn.Sequential(
                weather_cnn,
                torch.nn.Flatten(),
                torch.nn.Linear(np.prod(weather_cnn_features.shape[1:]), latent_weather_size),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm1d(latent_weather_size)
            )
        elif self.hparams.weather_architecture.lower() == '3dcnn':
            block = lambda input_size, features: [
                torch.nn.Conv3d(input_size, features, 3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2)
            ]
            weather_cnn = torch.nn.Sequential(
                # torch.nn.Upsample(size=(32, 32), mode='bilinear'),
                *block(1, 4),  # 32 -> 14
                *block(4, 8)  # 14 -> 5
            )
            weather_cnn_features = weather_cnn(example_weather[:, None, ...])
            self._weather_net = torch.nn.Sequential(
                weather_cnn,
                torch.nn.Flatten(),
                torch.nn.Linear(np.prod(weather_cnn_features.shape[1:]), latent_weather_size),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm1d(latent_weather_size)
            )
        elif self.hparams.weather_architecture.lower() == 'flatten':
            self._weather_net = torch.nn.Flatten()
        else:
            raise ValueError(f'Unkown weather architecture {self.hparams.weather_architecture}.')

    def _build_calendar_net(self, example_calendar, latent_calendar_size=32):
        # calendar processing part
        if self.hparams.calendar_architecture.lower() == 'fcn':
            self._calendar_net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(example_calendar.shape[1], latent_calendar_size),
                torch.nn.ReLU(),
                # torch.nn.BatchNorm1d(latent_calendar_size)
            )
        elif self.hparams.calendar_architecture.lower() == 'flatten':
            self._calendar_net = torch.nn.Sequential(
                torch.nn.Flatten(),
            )
        else:
            raise NotImplementedError(f'Unkown calendar architecture {self.hparams.calendar_architecture} not implemented.')

    def _forward_energy_batch(self, energy_batch):
        if self.hparams.energy_architecture.lower() == 'cnn':
            energy_batch = energy_batch.view(energy_batch.shape[0], 1, -1)
            forward_batch = self._energy_net(energy_batch)
        else:
            forward_batch = self._energy_net(energy_batch)
        return forward_batch

    def _forward_weather_batch(self, weather_batch):
        if self.hparams.weather_architecture.lower() == '3dcnn':
            weather_batch = weather_batch[:, None, ...]
        return self._weather_net(weather_batch)

    def _forward_calendar_batch(self, calendar_batch):
        return self._calendar_net(calendar_batch)
