import os
import glob

import calendar
import pprint

import numpy as np
import xarray as xr

import ecmwfapi


YEARS = np.arange(2015, 2020)
MONTHS = np.arange(1, 13)


def mars_request(request, target):
    pprint.pprint(request)
    client = ecmwfapi.ECMWFService('mars')
    client.execute(request, target)


def download_hres_grib_files(output_dir, years, months, resolution):
    for year in years:
        for month in months:
            last_day = calendar.monthrange(year, month)[1]
            params = [
                '167.128',  # temperature at 2m
                '165.128', '166.128',  # u, v component of wind at 10m
                '176.128',  # surface net solar radiation
                '210.128',  # surface net solar radiation (clear sky)
                '164.128',  # total cloud cover
                '228.128'  # total precipitation
            ]
            # set steps to [0, 3, ..., 21, 24, 27, ..., 165, 168]
            steps = np.arange(0, 144 + 1, 3)
            request = {
                'class': 'od',
                'date': f'{year}-{month:02}-01/to/{year}-{month:02}-{last_day:02}',
                'area': '56/4/46/17',
                'grid': f'{resolution}/{resolution}',
                'expver': '1',
                'levtype': 'sfc',
                'param': '/'.join(params),
                'step': '/'.join([f'{step:02}' for step in steps]),
                'stream': 'oper',
                'time': '00/12',
                'type': 'fc'
            }
            target_file = f'{year}_{month:02}.grib'
            mars_request(request, os.path.join(output_dir, target_file))


def merge_grib_files(input_dir, target):
    grib_files = glob.glob(os.path.join(input_dir, '*.grib'))
    grib_files = np.sort(grib_files)
    with open(target, 'wb') as outfile:
        for grib_file in grib_files:
            with open(grib_file, 'rb') as grib_file_obj:
                for line in grib_file_obj:
                    outfile.write(line)


def convert_to_netcdf(grib_file):
    ds = xr.open_dataset(grib_file, chunks={'time': 168})
    ds.to_netcdf(f'{os.path.splitext(grib_file)[0]}.nc')


if __name__ == '__main__':
    # os.makedirs('hres_10', exist_ok=True)
    # download_hres_grib_files('hres_10', YEARS, MONTHS, 0.1)
    merge_grib_files('hres_10', 'hres_10.grib')
    convert_to_netcdf('hres_10.grib')

    os.makedirs('hres_25', exist_ok=True)
    download_hres_grib_files('hres_25', YEARS, MONTHS, 0.25)
    merge_grib_files('hres_25', 'hres_25.grib')
    convert_to_netcdf('hres_25.grib')
