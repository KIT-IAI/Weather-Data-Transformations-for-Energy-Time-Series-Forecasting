from functools import partial
from multiprocessing import Pool

import numpy as np
import xarray as xr
from scipy.interpolate import griddata


def interpolate_single(ws_data, ws_longs, ws_lats, method):
    longs = np.arange(4, 17.01, 0.25)
    lats = np.arange(56, 45.99, -1 * 0.25)
    x, y = np.meshgrid(longs, lats)
    counts = np.full(x.shape, 0)
    data = np.full(x.shape, np.nan)
    for long, lat, z in zip(ws_longs, ws_lats, ws_data):
        pos_x = int((long - 4) / 0.25)
        pos_y = int((56 - lat) / 0.25)
        counts[pos_y, pos_x] += 1
        if np.isnan(data[pos_y, pos_x]):
            data[pos_y, pos_x] = z
        else:
            data[pos_y, pos_x] += z
    data = data / counts

    if method.lower() == 'nearest':
        interpolated_image = data
    else:
        mask = ~np.isnan(data)
        interpolated_image = griddata(
            list(zip(x[mask], y[mask])),
            data[mask],
            (x, y),
            method=method)
    nans = np.isnan(interpolated_image)
    mask = ~nans
    interpolated_image[nans] = griddata(
        list(zip(x[mask], y[mask])),
        interpolated_image[mask],
        (x, y),
        method='nearest')[nans]

    return interpolated_image


def interpolate_all(ds, key, method):
    with Pool() as pool:
        results = pool.map(partial(interpolate_single,
                                   ws_longs=ds.longitude.values,
                                   ws_lats=ds.latitude.values,
                                   method=method),
                           ds[key].values.T)
    return results


if __name__ == '__main__':
    weather_vars2keys = [
        ('temperature', 't2m'),
        ('solar', 'ssr'),
        ('wind', 'u+v')
    ]

    for method in ['nearest', 'linear', 'cubic']:
        ds_interpolated = xr.Dataset()
        for variable, key in weather_vars2keys:
            ds_ws = xr.load_dataset(f'{variable}.nc')
            results = interpolate_all(ds_ws, key, method)
            da_interpolated = xr.DataArray(
                data=np.array(results),
                dims=['time', 'latitude', 'longitude'],
                coords=dict(
                    time=ds_ws.time,
                    latitude=np.arange(56, 45.99, -1 * 0.25),
                    longitude=np.arange(4, 17.01, 0.25)
                )
            )
            ds_interpolated[key] = da_interpolated
        ds_interpolated.to_netcdf(f'{method}.nc')
