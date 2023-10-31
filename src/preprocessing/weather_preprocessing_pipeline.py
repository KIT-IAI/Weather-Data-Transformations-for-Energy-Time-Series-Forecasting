import os

from pywatts.core.pipeline import Pipeline
from pywatts.modules.wrappers import FunctionModule

import src.data_representations as dr
from src.pywatts import Normalizer


# Limits choosen by using web UI of ECMWF:
# https://apps.ecmwf.int/shopping-cart/orders/new
GEO_FILTER = {
    'germany': {
        'latitude': (46, 56),
        'longitude': (4, 17),
    },
    'transnetbw': {
        'latitude': (47, 50),
        'longitude': (7, 11),
    }
}


def get_geo_limits(hparams):
    latitude = GEO_FILTER[hparams.tso_target]['latitude']
    longitude = GEO_FILTER[hparams.tso_target]['longitude']
    return latitude, longitude


def filter_era5_data(hparams, input):
    latitude, longitude = get_geo_limits(hparams)
    if hparams.weather_source_param is not None and hparams.weather_source_param == 'dwd':
        filter_latitudes = (input.latitude >= latitude[0]) & \
                        (input.latitude <= latitude[1])
        filter_longitudes = (input.longitude >= longitude[0]) & \
                            (input.longitude <= longitude[1])
        return input.loc[:, :, filter_latitudes & filter_longitudes]
    else:
        # assume normal nwp data
        filter_latitudes = (input.latitude >= latitude[0]) & \
                        (input.latitude <= latitude[1])
        filter_longitudes = (input.longitude >= longitude[0]) & \
                            (input.longitude <= longitude[1])
        return input[:, :, filter_latitudes, filter_longitudes]


def filter_hres_data(hparams, input):
    latitude, longitude = get_geo_limits(hparams)
    if hparams.weather_source_param is not None and hparams.weather_source_param == 'dwd':
        filter_latitudes = (input.latitude >= latitude[0]) & \
                        (input.latitude <= latitude[1])
        filter_longitudes = (input.longitude >= longitude[0]) & \
                            (input.longitude <= longitude[1])
        return input.loc[:, filter_latitudes & filter_longitudes]
    else:
        # assume normal nwp data
        filter_latitudes = (input.latitude >= latitude[0]) & \
                        (input.latitude <= latitude[1])
        filter_longitudes = (input.longitude >= longitude[0]) & \
                            (input.longitude <= longitude[1])
        return input[:, filter_latitudes, filter_longitudes]


def filter_dwd_data(hparams, input):
    latitude, longitude = get_geo_limits(hparams)
    if hparams.weather_source_param is not None and hparams.weather_source_param in ['nearest', 'linear', 'cubic']:
        # assume interpolated dwd data
        filter_latitudes = (input.latitude >= latitude[0]) & \
                        (input.latitude <= latitude[1])
        filter_longitudes = (input.longitude >= longitude[0]) & \
                            (input.longitude <= longitude[1])
        return input[:, :, filter_latitudes, filter_longitudes]
    else:
        # assume normal weather station vector
        filter_latitudes = (input.latitude >= latitude[0]) & \
                        (input.latitude <= latitude[1])
        filter_longitudes = (input.longitude >= longitude[0]) & \
                            (input.longitude <= longitude[1])
        return input.loc[:, :, filter_latitudes & filter_longitudes]


def create_weather_preprocessing_pipeline(hparams):
    pipeline = Pipeline(path=os.path.join('run', 'preprocessing', 'weather'))

    # crop weather data depending on the selected TSO
    if hparams.weather_source.lower() == 'era5':
        cropped_weather_data = FunctionModule(
            lambda input: filter_era5_data(hparams, input), name='FilteredWeather')(input=pipeline['weather'])
    elif hparams.weather_source.lower() == 'hres':
        cropped_weather_data = FunctionModule(
            lambda input: filter_hres_data(hparams, input), name='FilteredWeather')(input=pipeline['weather'])
    elif hparams.weather_source.lower() == 'dwd':
        cropped_weather_data = FunctionModule(
            lambda input: filter_dwd_data(hparams, input), name='FilteredWeather')(input=pipeline['weather'])
    else:
        raise ValueError(f'Unkown weather source {hparams.weather_source}.')

    # weather data normalization
    weather_normalizer = Normalizer(method=hparams.scaling, name='WeatherNormalizer')
    normalized_weather_data = weather_normalizer(x=cropped_weather_data)

    # data representations
    if hparams.weather_source in ['dwd']:
        weather_class = 'dwd'
        if hparams.weather_source_param in ['nearest', 'linear', 'cubic']:
            # interpolated dwd data treated as nwp data
            weather_class = 'nwp'
    elif hparams.weather_source in ['era5', 'hres']:
        weather_class = 'nwp'
        if hparams.weather_source_param in ['dwd']:
            # nwp data treated as weather station data
            weather_class = 'dwd'

    if f'{weather_class}_{hparams.weather_data_representation_type}_fit' in dr.__dict__:
        weather_data_representation = FunctionModule(
                lambda energy, weather: getattr(dr, f'{weather_class}_{hparams.weather_data_representation_type}')(hparams, energy, weather),
                lambda energy, weather: getattr(dr, f'{weather_class}_{hparams.weather_data_representation_type}_fit')(hparams, energy, weather),
                name='WeatherDataRepresentation'
            )(energy=pipeline['energy'], weather=normalized_weather_data)
    else:
        weather_data_representation = FunctionModule(
                lambda energy, weather: getattr(dr, f'{weather_class}_{hparams.weather_data_representation_type}')(hparams, energy, weather),
                name='WeatherDataRepresentation'
            )(energy=pipeline['energy'], weather=normalized_weather_data)

    # weather data normalization
    weather_dr_normalizer = Normalizer(method=hparams.scaling, name='WeatherDataRepresentationNormalized')
    normalized_weather_dr = weather_dr_normalizer(x=weather_data_representation)

    return pipeline
