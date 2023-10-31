import numpy as np

from sklearn.decomposition import PCA

from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

global pca


def _parse_params(params):
    if 'k' in params:
        pca_components = params['k']
    else:
        pca_components = 0.95
    return pca_components


def nwp_pca_fit(hparams, energy_time_series, weather_time_series):
    """ Fit PCA data representation for weather time series. """
    global pca

    pca_components = _parse_params(hparams.weather_data_representation_params)

    pca = PCA(n_components=pca_components)
    weather_data = weather_time_series.values[:, 0]
    pca.fit(weather_data.reshape(weather_data.shape[0], -1))


def nwp_pca(hparams, energy_time_series, weather_time_series):
    """ Fit PCA data representation for weather time series. """
    global pca
    data = []
    for i in range(weather_time_series.shape[1]):
        weather_data = weather_time_series.values[:, i]
        new_data = pca.transform(weather_data.reshape(weather_data.shape[0], -1))
        data.append(new_data)
    data = np.swapaxes(np.array(data), 0, 1)
    return numpy_to_xarray(data, weather_time_series.time)
