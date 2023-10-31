import os

import numpy as np
import pandas as pd
import xarray as xr

from src.data_handling.dwd import load_dwd_data, split_dwd_data
from src.data_handling.era5 import load_era5_data, split_era5_data
from src.data_handling.hres import load_hres_data, split_hres_data


def load_data(hparams, data_dir='data'):
    """ Load dataset for the energy forecasting use case. """
    # load energy data
    if hparams.energy_class.lower() == 'load':
        if hparams.tso_target == 'germany':
            energy_df_key = f'DE_load_actual_entsoe_transparency'
        else:
            energy_df_key = f'DE_{hparams.tso_target.lower()}_load_actual_entsoe_transparency'
        energy_capacity_df_key = None
    elif hparams.energy_class.lower() == 'solar':
        if hparams.tso_target == 'germany':
            energy_df_key = 'DE_solar_generation_actual'
        else:
            energy_df_key = f'DE_{hparams.tso_target.lower()}_solar_generation_actual'
        energy_capacity_df_key = 'DE_solar_capacity'
    elif hparams.energy_class.lower() == 'wind':
        if hparams.tso_target == 'germany':
            energy_df_key = 'DE_wind_generation_actual'
        else:
            energy_df_key = f'DE_{hparams.tso_target.lower()}_wind_onshore_generation_actual'
        energy_capacity_df_key = 'DE_wind_capacity'

    df = pd.read_csv(os.path.join(data_dir, 'opsd', 'opsd.csv'))
    energy = df[energy_df_key]
    if energy_capacity_df_key is None:
        energy_capacity = np.ones(energy.shape)
    else:
        energy_capacity = df[energy_capacity_df_key]
        mask = np.isnan(energy_capacity)
        energy_capacity[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), energy_capacity[~mask])
    energy_ds = xr.Dataset(
        data_vars=dict(
            energy=(['time'], energy),
            energy_capacity=(['time'], energy_capacity),
        ),
        coords=dict(
            time=(['time'], pd.to_datetime(df['utc_timestamp']))
        )
    )

    # load weather data
    if hparams.weather_source.lower() == 'dwd':
        weather_ds = load_dwd_data(hparams, data_dir)
    elif hparams.weather_source.lower() == 'era5':
        weather_ds = load_era5_data(hparams, data_dir)
    elif hparams.weather_source.lower() == 'hres':
        weather_ds = load_hres_data(hparams, data_dir)
    else:
        raise KeyError(f'Unkown weather source {hparams.weather_source}.')

    # convert weather data to float32 to reduce memory usage
    for key in weather_ds.keys():
        weather_ds[key] = weather_ds[key].astype(np.float32)

    return dict(energy=energy_ds, weather=weather_ds)


def train_test_split(hparams, ds):
    """ Create train and test split. Train will also split into train and validation later. """
    # define train and test time range
    train_start = pd.to_datetime('2015-01-01 12:00')
    train_end = pd.to_datetime('2019-01-01 00:00')
    test_start = pd.to_datetime('2019-01-01 12:00')
    test_end = pd.to_datetime('2019-12-31 23:00')

    # get energy train and test data
    energy_train_idx = pd.date_range(start=train_start, end=train_end, freq='1h')
    energy_test_idx = pd.date_range(start=test_start, end=test_end, freq='1h')
    train_energy = ds['energy'].energy.loc[energy_train_idx]
    test_energy = ds['energy'].energy.loc[energy_test_idx]
    train_energy_capacity = ds['energy'].energy_capacity.loc[energy_train_idx]
    test_energy_capacity = ds['energy'].energy_capacity.loc[energy_test_idx]

    # get weather train and test data
    if hparams.weather_source == 'dwd' or hparams.weather_source_param == 'dwd':
        splits = split_dwd_data(
            hparams, ds,
            train_start, train_end,
            test_start, test_end
        )
    elif hparams.weather_source == 'era5':
        splits = split_era5_data(
            hparams, ds,
            train_start, train_end,
            test_start, test_end
        )
    elif hparams.weather_source == 'hres':
        splits = split_hres_data(
            hparams, ds,
            train_start, train_end,
            test_start, test_end
        )
    else:
        raise ValueError(f'Unkown weather source {hparams.weather_source}.')

    train_weather, test_weather = splits

    if 'number' in train_weather.coords:
        train_weather = train_weather.drop('number')
        test_weather = test_weather.drop('number')

    # prepare train data
    train = {
        'energy': train_energy,
        'energy_capacity': train_energy_capacity,
        'weather': train_weather,
    }

    # prepare test data
    test = {
        'energy': test_energy,
        'energy_capacity': test_energy_capacity,
        'weather': test_weather,
    }

    return train, test
