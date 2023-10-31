import numpy as np
import pandas as pd
import xarray as xr

from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


def _parse_params(params):
    if 'set' in params:
        stats_set = params['set'].lower()
    else:
        stats_set = 'small'
    if 'splits' in params:
        splits = params['splits'].split('x')
        splits = (int(splits[0]), int(splits[1]))
    else:
        splits = None
    if 'corr' in params:
        corr = params['corr']
    else:
        corr = None
    if 'width' in params:
        width = params['width']
    else:
        width = 168

    if stats_set == 'small':
        stats = ['mean']
    elif stats_set == 'medium':
        stats = ['mean', 'std']
    elif stats_set == 'large':
        stats = ['mean', 'std', 'min', 'max']

    return stats, splits, corr, width


def get_splits(weather_data, rows, cols):
    splits = list()
    for split_col in np.array_split(weather_data, cols, axis=3):
        for split_row in np.array_split(split_col, rows, axis=2):
            splits.append(split_row)
    return splits


def get_correlation(energy_time_series, weather_time_series, width=2160):
    correlation = np.zeros(weather_time_series.shape)
    x = pd.Series(energy_time_series)
    for i in range(weather_time_series.shape[1]):
        y = pd.Series(weather_time_series[:, i])
        correlation[:, i] = x.rolling(width).corr(y).values
    correlation[:width - 1] = correlation[width - 1]
    return correlation


def nwp_stats(hparams, energy_time_series, weather_time_series):
    """ Calculate statistical data representation for weather time series. """
    stats, splits, corr, width = _parse_params(hparams.weather_data_representation_params)
    if corr is not None:
        flatten_weather = weather_time_series.values.reshape(*weather_time_series.shape[:2], -1)
        correlation = get_correlation(energy_time_series.loc[weather_time_series.time],
                                      flatten_weather[:, 0], width=width)
        data = np.zeros((*weather_time_series.shape[:2], corr))
        args = correlation.argsort(axis=1)
        args = args[:, -1 * corr:]
        for i in range(flatten_weather.shape[1]):
            data[:, i] = np.take_along_axis(flatten_weather[:, i], args, axis=1)
    else:
        stat_results = list()
        for stat in stats:
            weather_data = weather_time_series.values
            if splits is None:
                flatten_weather = weather_data.reshape(*weather_data.shape[:2], -1)
                data = getattr(flatten_weather, stat)(axis=2)
                stat_results.append(data.reshape((*data.shape, 1)))
            else:
                weather_splits = get_splits(weather_data, *splits)
                for weather_split in weather_splits:
                    flatten_weather = weather_split.reshape((*weather_split.shape[:2], -1))
                    data = getattr(flatten_weather, stat)(axis=2)
                    stat_results.append(data.reshape((*data.shape, 1)))
        data = np.concatenate(stat_results, axis=2)

    return xr.DataArray(data, dims=['time', 'steps', 'stats'],
                        coords={
                            'time': weather_time_series.time,
                            'steps': weather_time_series.steps
                        })
