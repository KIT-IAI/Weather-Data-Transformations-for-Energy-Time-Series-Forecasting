from src.data_representations.nwp_stats import nwp_stats


def dwd_stats(hparams, energy_time_series, weather_time_series):
    return nwp_stats(hparams, energy_time_series, weather_time_series)
