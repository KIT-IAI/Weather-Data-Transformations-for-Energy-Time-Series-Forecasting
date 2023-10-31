from src.data_representations.nwp_autoencoder import nwp_autoencoder, nwp_autoencoder_fit


def dwd_autoencoder_fit(hparams, energy_time_series, weather_time_series):
    nwp_autoencoder_fit(hparams, energy_time_series, weather_time_series)


def dwd_autoencoder(hparams, energy_time_series, weather_time_series):
    return nwp_autoencoder(hparams, energy_time_series, weather_time_series)
