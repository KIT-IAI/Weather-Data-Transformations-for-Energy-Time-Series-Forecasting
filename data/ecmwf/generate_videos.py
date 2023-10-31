import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


def load_dataset(path):
    return xr.open_dataset(path)


def nwp2video(time, images, limits, title, fps=24, temp_dir='temp', output='output.mp4'):
    os.makedirs(temp_dir, exist_ok=True)
    for i in tqdm(range(len(images))):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.xaxis.set_ticks(range(5, 17, 2))
        ax.yaxis.set_ticks(range(47, 57, 2))
        pos = ax.imshow(images[i], interpolation='nearest', aspect='auto',
                  vmin=images.min(), vmax=images.max(), extent=limits)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pos, cax)
        ax.set_title(title)
        ax.set_xlabel(time[i].strftime('%d-%m-%y %H:%M'))
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir, f'image_{i:05}'))
        plt.close()
    path = os.path.join(temp_dir, 'image_%05d.png')
    os.system(f'ffmpeg -r {fps} -i {path} -c:v libx264 -y {output}')

if __name__ == '__main__':
    title = {
        't2m': 'Temperature at 2m [K]',
        'tp': 'Total Precipitation [m]',
        'tcc': 'Total Cloud Cover [0..1]',
        'cbh': 'Cloud Base Height [m]',
        'u10': 'U-Component of Netural Wind at 10m [m/s]',
        'v10': 'V-Component of Netural Wind at 10m [m/s]',
        'u+v': 'Speed of Netural Wind at 10m [m/s]',
        'ssr': 'Surface Solar Radiation Downwards [J/m²]',
        'ssrd': 'Surface Net Solar Radiation [J/m²]',
    }
    for key in title.keys():
        ds = load_dataset('era5_10.nc')
        if key == 'u+v':
            ds['u+v'] = np.sqrt(np.square(ds.u10) + np.square(ds.v10))
        limits = [
            ds.longitude.min(),
            ds.longitude.max(),
            ds.latitude.min(),
            ds.latitude.max()
        ]
        start_idx = np.where(ds.time == pd.to_datetime('2015-05-01'))[0][0]
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        nwp2video(pd.to_datetime(ds.time), ds[key][start_idx:start_idx + 24 * 7 * 12].values,
                limits=limits, title=title[key],
                temp_dir='temp', output=f'era5_10_{key}.mp4')

        ds = load_dataset('era5_25.nc')
        if key == 'u+v':
            ds['u+v'] = np.sqrt(np.square(ds.u10) + np.square(ds.v10))
        limits = [
            ds.longitude.min(),
            ds.longitude.max(),
            ds.latitude.min(),
            ds.latitude.max()
        ]
        start_idx = np.where(ds.time == pd.to_datetime('2015-05-01'))[0][0]
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        nwp2video(pd.to_datetime(ds.time), ds[key][start_idx:start_idx + 24 * 7 * 12].values,
                limits=limits, title=title[key],
                temp_dir='temp', output=f'era5_25_{key}.mp4')