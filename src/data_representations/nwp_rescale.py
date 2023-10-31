import numpy as np
import cv2

from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


def _parse_params(params):
    if 'r' in params:
        rescale_factor = params['r']
    else:
        rescale_factor = 8
    return rescale_factor


def nwp_rescale(hparams, energy_time_series, weather_time_series):
    """ Calculate naive data representation for weather time series. """

    data = weather_time_series.values
    rescale_factor = _parse_params(hparams.weather_data_representation_params)

    shape = (rescale_factor * np.array(data.shape[1:])).astype(int)
    scaling_method = lambda img: cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC)
    data = np.array(list(map(scaling_method, data)))

    return numpy_to_xarray(data, weather_time_series)
