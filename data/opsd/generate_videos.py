import os
import shutil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm


OPSD_PATH = 'opsd.csv'


def load_dataset(path):
    with open(path, 'r') as file:
        df = pd.read_csv(file)
    df.utc_timestamp = pd.to_datetime(df.utc_timestamp).dt.tz_convert(None)
    df = df.set_index('utc_timestamp')
    return df


def opsd2video(df, actual_key, timesteps, title, y_ticks, fps = 24, temp_dir='temp', output='output.mp4'):
    os.makedirs(temp_dir, exist_ok=True)
    for i, timestep in tqdm(list(enumerate(timesteps))):
        _, ax = plt.subplots(figsize=(4, 3))
        time_load = pd.date_range(timestep - pd.Timedelta(f'{168}h'), timestep, freq='1h')
        load = df.loc[time_load, actual_key] / 1000
        ax.plot(time_load, load, 'k')
        ax.set_title(title)
        ax.set_xlabel(timestep.strftime('%d-%m-%y %H:%M'))
        ax.set_ylabel('GW')
        ax.yaxis.set_ticks(range(*y_ticks))
        ax.tick_params(labelbottom=False)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir, f'image_{i:05}'))
        plt.close()
    path = os.path.join(temp_dir, 'image_%05d.png')
    os.system(f'ffmpeg -r {fps} -i {path} -c:v libx264 -y {output}')

if __name__ == '__main__':
    df = load_dataset(OPSD_PATH)
    timesteps = pd.date_range('2015-05-01', pd.to_datetime('2015-05-01') + pd.Timedelta(f'{24*7*12}h'), freq='1h')
    if os.path.exists('temp'):
        shutil.rmtree('temp')

    run_keys = [
        ('Load', 'DE_load_actual_entsoe_transparency', [30, 90, 10]),
        ('Solar', 'DE_solar_generation_actual', [-5, 35, 5]),
        ('Wind', 'DE_wind_generation_actual', [-5, 35, 5])
    ]
    
    for name, actual_key, y_ticks in run_keys:
        opsd2video(df, actual_key, timesteps, title=f'OPSD {name} Germany [GW]',
                   y_ticks=y_ticks, temp_dir='temp', output=f'opsd_{name.lower()}.mp4')
