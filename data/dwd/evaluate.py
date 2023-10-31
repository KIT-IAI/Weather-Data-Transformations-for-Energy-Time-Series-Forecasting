import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd
from shapely.geometry import Polygon

import matplotlib.pyplot as plt


def load_era5():
    ds = xr.load_dataset(os.path.join('..', 'ecmwf', 'era5_25.nc'))
    ds['u+v'] = np.sqrt(
        np.square(ds['u10']) + np.square(ds['v10'])
    )
    return ds


def get_cities_frame():
    cities_frame = pd.read_csv(os.path.join('..', '..', 'data', 'simple_maps', 'de.csv'))
    cities_frame = cities_frame.rename(columns={'lng': 'long'})
    cities_frame = gpd.GeoDataFrame(cities_frame, crs='epsg:4326', geometry=gpd.points_from_xy(cities_frame.long, cities_frame.lat))
    cities_frame = cities_frame.sort_values('population_proper', ascending=False)
    cities_frame = cities_frame.iloc[:10, :]
    return cities_frame


def get_germany_frame():
    germany = gpd.read_file(os.path.join('..', '..', 'data', 'geojson', 'counties_germany.geojson'))
    return germany


def get_world_frame():
    world = gpd.read_file(os.path.join('..', '..', 'data', 'geojson', 'europe.geojson'))
    return world


def get_grid():
    stride = 0.25
    cols = np.arange(4, 17 + stride, stride)
    rows = np.arange(46, 56 + stride, stride)
    rows = np.flip(rows)
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(Polygon([(x,y), (x+stride, y), (x+stride, y-stride), (x, y-stride)]))
    grid = gpd.GeoDataFrame({'geometry':polygons})
    return grid

def display_interpolations(key):
    ds_gt = load_era5()
    for method in ['nearest', 'linear', 'cubic']:
        ds_interpolated = xr.load_dataset(f'{method}.nc')
        for i in [12, 24 * 150 + 12]:
            plt.imshow(ds_gt[key].values[i])
            plt.tight_layout()
            plt.savefig(f'{key}_{i}_gt.png')
            plt.savefig(f'{key}_{i}_gt.pgf', backend='pgf')
            plt.close()

            plt.imshow(ds_interpolated[key].values[i])
            plt.tight_layout()
            plt.savefig(f'{key}_{i}_{method}.png')
            plt.savefig(f'{key}_{i}_{method}.pgf', backend='pgf')
            plt.close()


def display_weather_stations(key):
    weather_staions_data = xr.load_dataset(f'{key}.nc')
    dpi = 100
    fig, ax = plt.subplots(figsize=(530/dpi, 410/dpi), dpi=dpi)
    plt.xlim(4, 17)
    plt.ylim(46, 56)
    # ax.set_axis_off()

    world = get_world_frame()
    germany = get_germany_frame()
    cities = get_cities_frame()

    world.boundary.plot(
        ax=ax, color='black', edgecolor='black'
    )
    germany.plot(ax=ax, color='lightgrey', edgecolor='black')

    cities.plot(
        ax=ax, alpha=1.0, color='black', edgecolor='black'
    )
    grid = get_grid()
    for city in cities.city:
        filter_frame = cities.loc[cities.city == city]
        x_offset = 0.3
        y_offset = 0
        if city == 'Düsseldorf':
            x_offset = -2.65
            y_offset = -0.1
        if city == 'Köln':
            x_offset = -1.2
            y_offset = -0.5
        if city == 'Essen':
            x_offset = -1.5
            y_offset = 0.25
        plt.text(
            filter_frame.long.iloc[0] + x_offset, filter_frame.lat.iloc[0] + y_offset,
            filter_frame.city.iloc[0], bbox=dict(boxstyle='square,pad=0.1', fc='white', alpha=0.75)
        )

    latitude = (47, 50)
    longitude = (7, 11)
    filter_latitudes = (weather_staions_data.latitude >= latitude[0]) & \
                    (weather_staions_data.latitude <= latitude[1])
    filter_longitudes = (weather_staions_data.longitude >= longitude[0]) & \
                        (weather_staions_data.longitude <= longitude[1])
    mask = filter_longitudes & filter_latitudes

    geo_frame = gpd.GeoDataFrame(weather_staions_data.station_id[mask], crs='epsg:4326',
                                 geometry=gpd.points_from_xy(
                                    weather_staions_data.longitude[mask],
                                    weather_staions_data.latitude[mask]))
    geo_frame.plot(
        ax=ax, alpha=1.0, marker='x', color='blue', edgecolor='black'
    )
    geo_frame = gpd.GeoDataFrame(weather_staions_data.station_id[~mask], crs='epsg:4326',
                                 geometry=gpd.points_from_xy(
                                    weather_staions_data.longitude[~mask],
                                    weather_staions_data.latitude[~mask]))
    geo_frame.plot(
        ax=ax, alpha=1.0, marker='x', color='red', edgecolor='black'
    )
    grid.boundary.plot(ax=ax, alpha=0.0, color='k')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(f'{key}_map.png')
    plt.savefig(f'{key}_map.pgf', backend='pgf')
    plt.close()


if __name__ == '__main__':
    display_interpolations('t2m')
    display_interpolations('ssr')
    display_interpolations('u+v')

    display_weather_stations('temperature')
    display_weather_stations('solar')
    display_weather_stations('wind')