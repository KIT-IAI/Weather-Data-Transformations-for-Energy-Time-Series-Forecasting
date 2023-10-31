import pytz

import numpy as np
import pandas as pd
from workalendar import europe
import xarray as xr

import workalendar.europe as europe


def get_calendar_obj(hparams):
    if hparams.tso_target == 'transnetBW':
        return europe.BadenWurttemberg()
    else:
        return europe.Germany()


def calendar_features(hparams, energy_input, energy_target):
    """ Generate calendar features for each given time step. """
    forecast_time = pd.to_datetime(energy_target.time)
    timezone = pytz.timezone('Europe/Berlin')
    local_time = list(map(lambda x: timezone.localize(x), energy_target.time.to_series()))

    features = []
    num_steps = int(hparams.prediction_horizon / hparams.forecast_horizon)
    for _ in range(num_steps):
        if hparams.energy_class == 'load':
            calendar = get_calendar_obj(hparams)
            one_hot_encodings = [
                list(map(lambda x: x.tzname().lower() == 'cet',
                         local_time)),
                list(map(lambda x: x.tzname().lower() == 'cest',
                         local_time)),
                *[list(map(lambda x: calendar.is_holiday(x),
                         forecast_time - pd.Timedelta(f'{24 * i}h')))
                  for i in range(-1, 3)],
                *[(forecast_time + pd.Timedelta(f'{24 * i}h')).dayofweek >= 5
                  for i in range(-1, 3)]
            ]
            linear_encodings = [
                ((forecast_time.year - 2015) * 12 + forecast_time.month) / (5 * 12)
            ]
            sin_cos_encodings = [
                (forecast_time.hour, 24),
                (forecast_time.dayofweek, 7),
                ((forecast_time.month - 1), 11)
            ]
        else:
            one_hot_encodings = [
            ]
            linear_encodings = [
                ((forecast_time.year - 2015) * 12 + forecast_time.month - 1) / (5 * 12)
            ]
            sin_cos_encodings = [
                (forecast_time.hour, 24),
                ((forecast_time.dayofyear - 1), 365)
            ]

        step_features = one_hot_encodings
        step_features.extend(linear_encodings)
        for data, tmax in sin_cos_encodings:
            step_features.append(np.sin(2 * np.pi * data / tmax))
            step_features.append(np.cos(2 * np.pi * data / tmax))

        features.append(np.array(step_features, dtype=float).T)

        forecast_time = forecast_time + pd.Timedelta(f'{hparams.forecast_horizon}h')

    return xr.DataArray(
        data=np.array(features).swapaxes(0, 1),
        dims=['time', 'step', 'cal'],
        coords={
            'time': energy_target.time
        }
    )
