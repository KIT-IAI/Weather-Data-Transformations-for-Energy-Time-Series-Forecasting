import numpy as np

from sklearn.decomposition import PCA

from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

global pca


def _parse_params(params):
    if 'k' in params:
        pca_components = params['k']
    else:
        pca_components = 8


def energy_pca_fit(hparams, energy_time_series):
    """ Fit PCA data representation for energy time series. """
    global pca
    pca_components = _parse_params(hparams.energy_data_representation_params)
    pca = PCA(n_components=pca_components)
    energy_data = energy_time_series.values
    pca.fit(energy_data.reshape(energy_data.shape[0], -1))


def energy_pca(hparams, energy_time_series):
    """ Fit PCA data representation for energy time series. """
    global pca
    energy_data = energy_time_series.values
    data = pca.transform(energy_data.reshape(energy_data.shape[0], -1))
    return numpy_to_xarray(data, energy_time_series.time)
