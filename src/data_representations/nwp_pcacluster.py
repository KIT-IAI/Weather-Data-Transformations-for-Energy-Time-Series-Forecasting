import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer

from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


global pca, kmeans


def _parse_params(params):
    if 'k' in params:
        pca_components = params['k']
    else:
        pca_components = 0.98

    if 'c' in params:
        cluster_components = params['c']
    else:
        cluster_components = 5

    if 'type' in params:
        representation = params['type']
    else:
        representation = 'distance'

    return pca_components, cluster_components, representation


def nwp_pcacluster_fit(hparams, energy_time_series, weather_time_series):
    """ Fit PCA data representation for weather time series. """
    global pca, kmeans, encoder

    pca_components, cluster_components, _ = \
        _parse_params(hparams.weather_data_representation_params)

    weather_data = weather_time_series.values[:, 0]
    pca = PCA(n_components=pca_components)
    kmeans = KMeans(n_clusters=cluster_components)
    encoder = LabelBinarizer()

    pca_transform = pca.fit_transform(weather_data.reshape(weather_data.shape[0], -1))
    kmeans.fit(pca_transform)
    clusters = kmeans.predict(pca_transform)
    encoder.fit(clusters)


def nwp_pcacluster(hparams, energy_time_series, weather_time_series):
    """ Fit PCA data representation for weather time series. """
    global pca, kmeans, encoder

    _, _, representation = \
        _parse_params(hparams.weather_data_representation_params)

    data = []
    for i in range(weather_time_series.shape[1]):
        weather_data = weather_time_series.values[:, i]
        new_data = pca.transform(weather_data.reshape(weather_data.shape[0], -1))

        if representation == 'onehot':
            new_data = kmeans.predict(new_data)
            new_data = encoder.transform(new_data)
        elif representation == 'distance':
            new_data = kmeans.transform(new_data)
        else:
            raise ValueError(f'Unkown representation type {representation}.')

        data.append(new_data)

    data = np.swapaxes(np.array(data), 0, 1)

    return numpy_to_xarray(data, weather_time_series.time)
