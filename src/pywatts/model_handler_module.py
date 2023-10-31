import os
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sqlalchemy import over

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

from pywatts.core.base import BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from src.pytorch_lightning import TerminalCallback
from src.models.forecasting import Forecasting


class MyDataset(Dataset):
    """
    Dataset class to return tuple data for multiple input networks.
    """
    def __init__(self, hparams, energy, weather, calendar, y=None, is_test=False):
        """ Initialize energy, calendar, and target datasets. """
        self.hparams = hparams
        self.energy = energy.astype(np.float32)
        self.weather = weather.astype(np.float32)
        self.calendar = calendar.astype(np.float32)
        self.is_test = is_test
        if y is None:
            self.y = None
        else:
            self.y = y.astype(np.float32)

    def __getitem__(self, index):
        """ Get tuple data for multiple input networks. """
        energy = self.energy[index].flatten()
        weather = self.weather[index]
        calendar = self.calendar[index].flatten()

        if self.hparams.energy_augmentations and not self.is_test:
            noise = np.random.normal(size=energy.shape, scale=0.025) \
                        .astype(np.float32)
            energy = energy + noise

        if self.hparams.weather_augmentations and not self.is_test:
            noise = np.random.normal(size=weather.shape, scale=0.025) \
                        .astype(np.float32)
            weather = weather + noise

        if self.y is None:
            return (energy, weather, calendar)
        else:
            y = self.y[index]
            return (energy, weather, calendar), self.y[index]

    def __len__(self):
        """ Get the length of the dataset. """
        return len(self.energy)


class ModelHandler(BaseEstimator):
    """
    PyWATTS model handler class to initialize, train, and predict neural network models.
    """

    def __init__(self, hparams, name: str = "DNN"):
        super().__init__(name)
        self.hparams = hparams
        self.models = []
        self.trainers = []

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def fit(self, energy, weather, calendar, y):
        """ Train, validate, and test the model based on hyperparameters set. """
        time = energy.time
        split_time = pd.to_datetime(self.hparams.train_split)
        train_idx = time < split_time

        inferences = []
        input_energy = energy.values
        for step in range(self.hparams.num_models):
            input_weather = weather.loc[:, step * self.hparams.forecast_horizon:(step + 1) * self.hparams.forecast_horizon].values
            input_calendar = calendar.loc[:, step].values
            train = MyDataset(
                self.hparams,
                input_energy[train_idx],
                input_weather[train_idx],
                input_calendar[train_idx],
                y.loc[train_idx, step * self.hparams.forecast_horizon + 1:(step + 1) * self.hparams.forecast_horizon].values,
            )
            validation = MyDataset(
                self.hparams,
                input_energy[~train_idx],
                input_weather[~train_idx],
                input_calendar[~train_idx],
                y.loc[~train_idx, step * self.hparams.forecast_horizon + 1:(step + 1) * self.hparams.forecast_horizon].values,
                is_test=True
            )

            train_loader = DataLoader(train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(validation, batch_size=self.hparams.batch_size, num_workers=0)

            self.train_model(train_loader, val_loader, step=step)


            dataset = MyDataset(self.hparams, input_energy, input_weather, input_calendar, is_test=True)
            data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=0)
            inference = self.trainers[step].predict(
                ckpt_path=self.trainers[step].checkpoint_callback.best_model_path,
                dataloaders=data_loader
            )
            inference = torch.concat(inference).cpu().numpy()
            inferences.append(inference)
            input_energy = np.concatenate(
                [energy[:, (-168 + (step + 1) * self.hparams.forecast_horizon):].values,
                    np.concatenate(inferences, axis=1)],
                    axis=1
            )

        self.is_fitted = True

    def train_model(self, train_loader, val_loader, step=0):
        # init loggers and trainer
        terminal_callback = TerminalCallback()
        if wandb.run.name is not None:
            ckpt_path = os.path.join('ckpts', wandb.run.name + '_' + wandb.run.id, f'{step:02}')
        else:
            ckpt_path = os.path.join('ckpts', f'{step:02}')
        callbacks = [
            terminal_callback,
            ModelCheckpoint(monitor=self.hparams.monitor, dirpath=ckpt_path, save_top_k=1),
            EarlyStopping(monitor=self.hparams.monitor, **self.hparams.early_stopping_params)
        ]

        wandb_logger = WandbLogger()
        logger = [
            wandb_logger
        ]

        if step > 0:
            self.models.append(
                Forecasting.load_from_checkpoint(
                    self.trainers[step - 1].checkpoint_callback.best_model_path,
                    dataset=next(iter(train_loader))
                )
            )
        else:
            self.models.append(Forecasting(self.hparams, next(iter(train_loader))))

        self.trainers.append(
            Trainer(gpus=1, callbacks=callbacks, logger=logger, max_epochs=1000,
                    enable_progress_bar=False))

        # learning rate finder
        # lr_finder = self.trainers[step].tuner.lr_find(self.models[step], min_lr=1e-10, max_lr=1e-1, num_training=1000,
        #                                               train_dataloaders=train_loader, val_dataloaders=val_loader)
        # fig = lr_finder.plot(suggest=True)
        # plt.savefig('lr_finder.png')
        # plt.close()
        # print(lr_finder.suggestion())
        # exit()

        # train, evaluate and test model
        self.trainers[step].fit(self.models[step], train_dataloaders=train_loader, val_dataloaders=val_loader)

        terminal_callback.plot_metric(keys=['loss', 'val_loss'],
                                      filename=f'train_val_{step}.png')

    def transform(self, energy, weather, calendar, y):
        """ Forecast energy based on the trained network. """

        for model in self.models:
            model.eval()

        with torch.no_grad():
            # predict data to return for train, validation, and test
            inferences = []
            inference = np.array([])
            num_steps = int(self.hparams.prediction_horizon / self.hparams.forecast_horizon)
            input_energy = energy.values
            for step in range(num_steps):
                input_weather = weather.loc[:, step * self.hparams.forecast_horizon:(step + 1) * self.hparams.forecast_horizon].values
                input_calendar = calendar.loc[:, step].values
                dataset = MyDataset(self.hparams, input_energy, input_weather, input_calendar, is_test=True)
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=0)
                if len(self.trainers) > step:
                    inference = self.trainers[step].predict(
                        ckpt_path=self.trainers[step].checkpoint_callback.best_model_path,
                        dataloaders=data_loader
                    )
                else:
                    inference = self.trainers[-1].predict(
                        ckpt_path=self.trainers[-1].checkpoint_callback.best_model_path,
                        dataloaders=data_loader
                    )
                inference = torch.concat(inference).cpu().numpy()
                inferences.append(inference)
                input_energy = np.concatenate(
                    [energy[:, (-168 + (step + 1) * self.hparams.forecast_horizon):].values,
                     np.concatenate(inferences, axis=1)],
                     axis=1
                )

            # perform model test on test dataset
            is_test = (energy.time == pd.to_datetime(self.hparams.test_split)).any()
            if is_test:
                # test
                dataset = MyDataset(
                    self.hparams,
                    energy.values,
                    weather.loc[:, :self.hparams.forecast_horizon].values,
                    calendar.loc[:, 0].values,
                    y.loc[:, :self.hparams.forecast_horizon].values,
                    is_test=True
                )
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=0)
                self.trainers[0].test(
                    ckpt_path=self.trainers[0].checkpoint_callback.best_model_path,
                    dataloaders=data_loader
                )

        inference = np.concatenate(inferences, axis=1)
        return numpy_to_xarray(inference, energy)