import os

import numpy as np
import pandas as pd
import xarray as xr

from src.data_handling.dwd import load_dwd_data


def load_hres_data(hparams, data_dir):
    if hparams.weather_source_param is None or hparams.weather_source_param == 'dwd':
        weather_ds = xr.load_dataset(os.path.join(data_dir, 'ecmwf', f'hres_25.nc'))
        weather_ds['u+v'] = np.sqrt(np.square(weather_ds.u10) + np.square(weather_ds.v10))
    else:
        weather_ds = xr.load_dataset(os.path.join(data_dir, 'ecmwf', f'hres_{hparams.weather_source_param}.nc'))
        weather_ds['u+v'] = np.sqrt(np.square(weather_ds.u10) + np.square(weather_ds.v10))

    if hparams.weather_source_param is not None and hparams.weather_source_param == 'dwd':
        weather_dict = dict()
        mapping = {
            'temperature': 't2m',
            'solar': 'ssr',
            'wind': 'u+v'
        }
        dwd_ds = load_dwd_data(hparams, data_dir)
        for weather_variable in ['temperature', 'solar', 'wind']:
            longs = dwd_ds[weather_variable].longitude.values
            lats = dwd_ds[weather_variable].latitude.values
            xs = ((longs - 4) / 0.25).astype(int)
            ys = ((56 - lats) / 0.25).astype(int)
            target_time = weather_ds.step == pd.to_timedelta(f'{hparams.forecast_horizon}h')
            weather_data = weather_ds[mapping[weather_variable]].values[:, target_time, ys, xs]
            weather_data = weather_data.reshape(-1, len(dwd_ds[weather_variable].station_id)).T
            data_vars = {
                'latitude': (['station_id'], lats),
                'longitude': (['station_id'], longs),
                mapping[weather_variable]: (
                    ['station_id', 'time'],
                    weather_data
                )
            }
            weather_dict[weather_variable] = xr.Dataset(
                data_vars=data_vars,
                coords=dict(
                    station_id=(['station_id'], dwd_ds[weather_variable].station_id.values),
                    time=(['time'], weather_ds.time.values + weather_ds.step[target_time].values[0])
                )
            )
        weather_ds = weather_dict

    return weather_ds


def split_hres_data(hparams, ds,
                    train_start, train_end,
                    test_start, test_end,
                    forecasting_freq):
    # ECMWF HRES dataset offering weather forecasts for midday and midnight
    weather_train_idx = pd.date_range(start=train_start, end=train_end, freq=forecasting_freq)
    weather_test_idx = pd.date_range(start=test_start, end=test_end, freq=forecasting_freq)
    step_idx = pd.Timedelta(f'{hparams.forecast_horizon}h')
    train_temp = ds['weather']['t2m'].loc[
        weather_train_idx, ds['weather'].step.loc[step_idx]]
    train_temp = train_temp.assign_coords(time=weather_train_idx + step_idx)
    train_solar = ds['weather']['ssr'].loc[
        weather_train_idx, ds['weather'].step.loc[step_idx]]
    train_solar = train_solar.assign_coords(time=weather_train_idx + step_idx)
    train_wind = ds['weather']['u+v'].loc[weather_train_idx, ds['weather'].step.loc[step_idx]]
    train_wind = train_wind.assign_coords(time=weather_train_idx + step_idx)

    test_temp = ds['weather']['t2m'].loc[
        weather_test_idx, ds['weather'].step.loc[step_idx]]
    test_temp = test_temp.assign_coords(time=weather_test_idx + step_idx)
    test_solar = ds['weather']['ssr'].loc[
        weather_test_idx, ds['weather'].step.loc[step_idx]]
    test_solar = test_solar.assign_coords(time=weather_test_idx + step_idx)
    test_wind = ds['weather']['u+v'].loc[weather_test_idx, ds['weather'].step.loc[step_idx]]
    test_wind = test_wind.assign_coords(time=weather_test_idx + step_idx)

    return (
        (train_temp, train_solar, train_wind),
        (test_temp, test_solar, test_wind)
    )