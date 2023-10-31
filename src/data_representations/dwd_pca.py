from src.data_representations.nwp_pca import nwp_pca, nwp_pca_fit


def dwd_pca_fit(hparams, energy_time_series, weather_time_series):
    nwp_pca_fit(hparams, energy_time_series, weather_time_series)


def dwd_pca(hparams, energy_time_series, weather_time_series):
    return nwp_pca(hparams, energy_time_series, weather_time_series)
