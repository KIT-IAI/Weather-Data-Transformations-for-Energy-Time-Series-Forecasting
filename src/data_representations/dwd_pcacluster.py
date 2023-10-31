from src.data_representations.nwp_pcacluster import nwp_pcacluster, nwp_pcacluster_fit


def dwd_pcacluster_fit(hparams, energy_time_series, weather_time_series):
    nwp_pcacluster_fit(hparams, energy_time_series, weather_time_series)


def dwd_pcacluster(hparams, energy_time_series, weather_time_series):
    return nwp_pcacluster(hparams, energy_time_series, weather_time_series)
