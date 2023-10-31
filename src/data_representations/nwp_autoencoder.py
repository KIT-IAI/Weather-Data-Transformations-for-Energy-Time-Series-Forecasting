import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from src.models.autoencoder import FCNAutoencoder, CNNAutoencoder
from src.pytorch_lightning import TerminalCallback


global autoencoder, data_stats


def prepare_dateloader(hparams, input_data):
    input_data = input_data.reshape(-1, 1, *input_data.shape[1:])
    dataset = TensorDataset(torch.Tensor(input_data))
    data_loader = DataLoader(dataset, batch_size=256)
    return data_loader


def nwp_autoencoder_fit(hparams, energy_time_series, weather_time_series):
    global autoencoder, data_stats

    if hparams.weather_data_representation_params:
        model_class = hparams.weather_data_representation_params['model'].lower()
    else:
        model_class = 'fcn'

    if model_class == 'fcn':
        model = FCNAutoencoder
    elif model_class == 'cnn':
        model = CNNAutoencoder
    else:
        raise ValueError(f'Unkown model param {model_class} in autoencoder data representation.')

    pl.seed_everything(hparams.seed)
    autoencoder = model(hparams, weather_time_series[0, 0].values)

    train_idx = weather_time_series.time < pd.to_datetime(hparams.train_split)

    train = weather_time_series.loc[train_idx, 0].values
    train_dl = prepare_dateloader(hparams, train)
    val = weather_time_series.loc[~train_idx, 0].values
    val_dl = prepare_dateloader(hparams, val)

    terminal_callback = TerminalCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('ckpts', 'autoencoder'),
        filename=f'{datetime.now().strftime("%Y-%m-%d_%H:%m:%S")}_autoencoder',
        save_top_k=1
    )
    callbacks = [
        terminal_callback,
        checkpoint_callback,
        EarlyStopping('val_loss', patience=10, min_delta=0.001)
    ]
    trainer = pl.Trainer(gpus=-1, max_epochs=250,
                         callbacks=callbacks, progress_bar_refresh_rate=0)
    trainer.fit(autoencoder, train_dataloaders=train_dl, val_dataloaders=val_dl)
    terminal_callback.plot_metric(keys=['train_loss', 'val_loss'],
                                  filename='train_val_autoenc.png')

    autoencoder = model.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        hparams=hparams, example_input=weather_time_series[0, 0].values
    )


def nwp_autoencoder(hparams, energy_time_series, weather_time_series):
    global autoencoder, data_stats
    with torch.no_grad():
        data = []
        for i in range(weather_time_series.shape[1]):
            input_data = weather_time_series.values[:, i]
            input_data = input_data.reshape(-1, 1, *input_data.shape[1:])
            new_data = autoencoder.encode(torch.Tensor(input_data))
            new_data = new_data.cpu().numpy()
            data.append(new_data)
        data = np.swapaxes(np.array(data), 0, 1)
    return numpy_to_xarray(data, weather_time_series.time)
