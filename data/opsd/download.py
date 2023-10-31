import os
import requests

import pandas as pd
import matplotlib.pyplot as plt


OPSD_URL = 'https://data.open-power-system-data.org/time_series/2020-10-06/time_series_15min_singleindex.csv'
OPSD_KEEP = [
    'DE_load_actual_entsoe_transparency',
    'DE_load_forecast_entsoe_transparency',
    'DE_solar_capacity',
    'DE_solar_generation_actual',
    'DE_solar_profile',
    'DE_wind_capacity',
    'DE_wind_generation_actual',
    'DE_wind_profile',
    'DE_wind_offshore_capacity',
    'DE_wind_offshore_generation_actual',
    'DE_wind_offshore_profile',
    'DE_wind_onshore_capacity',
    'DE_wind_onshore_generation_actual',
    'DE_wind_onshore_profile',
    'DE_50hertz_load_actual_entsoe_transparency',
    'DE_50hertz_load_forecast_entsoe_transparency',
    'DE_50hertz_solar_generation_actual',
    'DE_50hertz_wind_generation_actual',
    'DE_50hertz_wind_offshore_generation_actual',
    'DE_50hertz_wind_onshore_generation_actual',
    'DE_LU_load_actual_entsoe_transparency',
    'DE_LU_load_forecast_entsoe_transparency',
    'DE_LU_solar_generation_actual',
    'DE_LU_wind_generation_actual',
    'DE_LU_wind_offshore_generation_actual',
    'DE_LU_wind_onshore_generation_actual',
    'DE_amprion_load_actual_entsoe_transparency',
    'DE_amprion_load_forecast_entsoe_transparency',
    'DE_amprion_solar_generation_actual',
    'DE_amprion_wind_onshore_generation_actual',
    'DE_tennet_load_actual_entsoe_transparency',
    'DE_tennet_load_forecast_entsoe_transparency',
    'DE_tennet_solar_generation_actual',
    'DE_tennet_wind_generation_actual',
    'DE_tennet_wind_offshore_generation_actual',
    'DE_tennet_wind_onshore_generation_actual',
    'DE_transnetbw_load_actual_entsoe_transparency',
    'DE_transnetbw_load_forecast_entsoe_transparency',
    'DE_transnetbw_solar_generation_actual',
    'DE_transnetbw_wind_onshore_generation_actual'
]


def download(url, output):
    r = requests.get(url)
    with open(output, "wb") as file:
        file.write(r.content)



def load_csv(path):
    with open(path, 'rb') as file:
        opsd = pd.read_csv(file)
        file.close()
    opsd['utc_timestamp'] = pd.to_datetime(opsd['utc_timestamp'])
    opsd.set_index('utc_timestamp', inplace=True)
    return opsd


def resample(opsd):
    for key in opsd:
        opsd[key].resample()


def main():
    # download and load original opsd file
    # download(OPSD_URL, 'opsd_orig.csv')
    opsd = load_csv('opsd_orig.csv')

    # filter needed columns
    opsd = opsd[OPSD_KEEP]

    # resample to hourly resolution
    opsd = opsd.resample('1h').mean()
    opsd.index = opsd.index + pd.to_timedelta('1h')

    # save opsd csv file
    opsd.to_csv('opsd.csv')

if __name__ == '__main__':
    main()
