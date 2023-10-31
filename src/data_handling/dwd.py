import os

import numpy as np
import pandas as pd
import xarray as xr


def load_dwd_data(hparams, data_dir):
    weather_ds = {
        't2m': xr.load_dataset(
            os.path.join(data_dir, 'dwd', 'temperature.nc')),
        'ssr': xr.load_dataset(
            os.path.join(data_dir, 'dwd', 'solar.nc')),
        'u+v': xr.load_dataset(
            os.path.join(data_dir, 'dwd', 'wind.nc'))
    }

    if hparams is not None and hparams.weather_source_param is not None:
        # 'nearest', 'linear', and 'cubic' should be
        # available as interpolation types
        interpolation_type = hparams.weather_source_param.lower()
        path = os.path.join(data_dir, 'dwd', f'{interpolation_type}.nc')
        weather_ds = xr.load_dataset(path)

    return weather_ds


def split_dwd_data(hparams, ds,
                   train_start, train_end,
                   test_start, test_end):

    time = ds['weather'][hparams.weather_variable].time
    time_deltas = np.arange(0, hparams.prediction_horizon + 1,
                            step=hparams.weather_freq)
    weather_train_idx = []
    weather_test_idx = []
    for time_delta in time_deltas:
        time_delta = pd.Timedelta(f'{time_delta}h')
        train_time = pd.date_range(start=train_start + time_delta,
                                   end=train_end, freq=f'{hparams.forecast_freq}h')
        test_time = pd.date_range(start=test_start + time_delta,
                                  end=test_end, freq=f'{hparams.forecast_freq}h')
        _, train_time_idx, _ = np.intersect1d(time, train_time, return_indices=True)
        _, test_time_idx, _ = np.intersect1d(time, test_time, return_indices=True)
        weather_train_idx.append(train_time_idx.reshape(1, -1))
        weather_test_idx.append(test_time_idx.reshape(1, -1))
    min_train_idx = [x.shape[1] for x in weather_train_idx][-1]
    weather_train_idx = [x[:, :min_train_idx] for x in weather_train_idx]
    weather_train_idx = np.concatenate([x.reshape(1, -1) for x in weather_train_idx])
    min_test_idx = [x.shape[1] for x in weather_test_idx][-1]
    weather_test_idx = [x[:, :min_test_idx] for x in weather_test_idx]
    weather_test_idx = np.concatenate([x.reshape(1, -1) for x in weather_test_idx])

    if hparams.weather_source_param is None or hparams.weather_source_param == 'dwd':
        # one dimensional weather station vector
        data = ds['weather'][hparams.weather_variable][hparams.weather_variable] \
                    .T.values[weather_train_idx.T, :]
        train_weather = xr.DataArray(
            data=data,
            coords={
                'time': ds['weather'][hparams.weather_variable]
                            .time[weather_train_idx[0]],
                'steps': time_deltas,
                'station_id': ds['weather'][hparams.weather_variable].station_id,
                'latitude': ds['weather'][hparams.weather_variable].latitude,
                'longitude': ds['weather'][hparams.weather_variable].longitude
            },
            dims=['time', 'steps', 'station_id']
        )

        data = ds['weather'][hparams.weather_variable][hparams.weather_variable] \
                    .T.values[weather_test_idx.T, :]
        test_weather = xr.DataArray(
            data=data,
            coords={
                'time': ds['weather'][hparams.weather_variable]
                            .time[weather_test_idx[0]],
                'steps': time_deltas,
                'station_id': ds['weather'][hparams.weather_variable].station_id,
                'latitude': ds['weather'][hparams.weather_variable].latitude,
                'longitude': ds['weather'][hparams.weather_variable].longitude
            },
            dims=['time', 'steps', 'station_id']
        )
    else:
        # interpolated weather station data
        data = ds['weather'][hparams.weather_variable].values[weather_train_idx, :]
        data = data.swapaxes(0, 1)
        train_weather = xr.DataArray(
            data=data,
            coords={
                'time': ds['weather'][hparams.weather_variable].time[weather_train_idx[0]],
                'steps': time_deltas,
                'latitude': ds['weather'][hparams.weather_variable].latitude,
                'longitude': ds['weather'][hparams.weather_variable].longitude
            },
            dims=['time', 'steps', 'latitude', 'longitude']
        )

        data = ds['weather'][hparams.weather_variable].values[weather_test_idx, :]
        data = data.swapaxes(0, 1)
        test_weather = xr.DataArray(
            data=data,
            coords={
                'time': ds['weather'][hparams.weather_variable].time[weather_test_idx[0]],
                'steps': time_deltas,
                'latitude': ds['weather'][hparams.weather_variable].latitude,
                'longitude': ds['weather'][hparams.weather_variable].longitude
            },
            dims=['time', 'steps', 'latitude', 'longitude']
        )

    return train_weather, test_weather