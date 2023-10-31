import numpy as np

from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


def energy_differences(hparams, energy_time_series):
    """ Calculate differences depending on the forecast horizon. """
    # transform xarray data array to numpy array
    data = energy_time_series.values

    # prepare difference kernel to convolve
    kernel = np.zeros(hparams.forecast_horizonforecast_horizon + 1)
    kernel[0] = 1
    kernel[-1] = -1

    # apply kernel to time series data
    data = np.convolve(data, kernel, mode='valid')

    return numpy_to_xarray(data, energy_time_series[hparams.forecast_horizon:])
