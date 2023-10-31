import os
import glob

import calendar
import pprint

import numpy as np

import cdsapi


YEARS = np.arange(2015, 2020)
MONTHS = np.arange(1, 13)


def cds_request(request, target):
    client = cdsapi.Client()
    pprint.pprint(request)
    pprint.pprint(target)
    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        target
    )


def download_era5_grib_files(output_dir, years, months, resolution):
    for year in years:
        for month in months:
            num_days = calendar.monthrange(year, month)[1]
            days = [f'{year}-{month:02}-{day + 1:02}' for day in range(num_days)]
            variables = [
                    '2m_temperature',
                    'total_precipitation',
                    'total_cloud_cover',
                    'cloud_base_height',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'surface_net_solar_radiation',
                    'surface_solar_radiation_downwards',
                ]
            request = {
                'product_type': 'reanalysis',
                'date': days,
                'variable': variables,
                'time': [f'{i:02}:00:00' for i in range(0, 24)],
                'area': '56/4/46/17',
                'grid': f'{resolution}/{resolution}',
                # Using netcdf instead of grib because for grib files
                # the grid size of the input parameters do not match
                'format': 'netcdf',
            }
            file_name = f'{year}_{month:02}.nc'
            cds_request(request, os.path.join(output_dir, file_name))


if __name__ == '__main__':
    os.makedirs('era5_25', exist_ok=True)
    download_era5_grib_files('era5_25', YEARS, MONTHS, 0.25)

    os.makedirs('era5_10', exist_ok=True)
    download_era5_grib_files('era5_10', YEARS, MONTHS, 0.1)
