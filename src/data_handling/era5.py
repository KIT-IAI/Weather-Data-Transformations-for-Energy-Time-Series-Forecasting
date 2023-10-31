import os

import numpy as np
import pandas as pd
import xarray as xr

from src.data_handling.dwd import load_dwd_data


def load_era5_data(hparams, data_dir):
    if hparams.weather_source_param is None or hparams.weather_source_param == 'dwd':
        weather_ds = xr.load_dataset(os.path.join(data_dir, 'ecmwf', f'era5_25.nc'))
        weather_ds['u+v'] = np.sqrt(np.square(weather_ds.u10) + np.square(weather_ds.v10))
    else:
        weather_ds = xr.load_dataset(os.path.join(data_dir, 'ecmwf', f'era5_{hparams.weather_source_param}.nc'))
        weather_ds['u+v'] = np.sqrt(np.square(weather_ds.u10) + np.square(weather_ds.v10))

    if hparams.weather_source_param is not None and hparams.weather_source_param == 'dwd':
        weather_dict = dict()
        dwd_ds = load_dwd_data(None, data_dir)
        for weather_variable in ['t2m', 'ssr', 'u+v']:
            longs = dwd_ds[weather_variable].longitude.values
            lats = dwd_ds[weather_variable].latitude.values
            xs = ((longs - 4) / 0.25).astype(int)
            ys = ((56 - lats) / 0.25).astype(int)
            data_vars = {
                'latitude': (['station_id'], lats),
                'longitude': (['station_id'], longs),
                weather_variable: (
                    ['station_id', 'time'],
                    weather_ds[weather_variable].values[:, ys, xs].T
                )
            }
            weather_dict[weather_variable] = xr.Dataset(
                data_vars=data_vars,
                coords=dict(
                    station_id=(['station_id'], dwd_ds[weather_variable].station_id.values),
                    time=(['time'], weather_ds.time.values)
                )
            )
        weather_ds = weather_dict


    return weather_ds


def split_era5_data(hparams, ds,
                    train_start, train_end,
                    test_start, test_end):
    # NOTE: Due to comparability, the ERA5 data in the same shape as HRES data.
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