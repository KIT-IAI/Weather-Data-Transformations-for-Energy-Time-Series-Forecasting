import os

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.pipeline import Pipeline
from pywatts.modules.wrappers import FunctionModule
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from src.preprocessing.energy_preprocessing_pipeline import create_energy_preprocessing_pipeline
from src.preprocessing.target_preprocessing_pipeline import create_target_preprocessing_pipeline
from src.preprocessing.weather_preprocessing_pipeline import create_weather_preprocessing_pipeline
import src.data_representations as dr


def validate(input):
    """
    Perform validation checks before starting pipeline to ensure only
    valid data is processed.
    """
    nan_idx = np.where(np.isnan(input.values))[0]
    if len(nan_idx) > 0:
        print('Found nan values!')
        print(input.time[nan_idx])
        raise ValueError('Energy time-series containing nan values.')
    return input


def reorder(input, base):
    shape = base.values.shape
    reordered_data = input.values.reshape(shape)
    return numpy_to_xarray(reordered_data, base)


def reshape_weather(input):
    data = input.values
    if len(data.shape) > 2:
        new_shape = [data.shape[0], 1, *data.shape[1:]]
        return numpy_to_xarray(data.reshape(new_shape), input)
    else:
        return input


def filter_data_representations(hparams, energy_data_representation, weather_data_representation):
    energy_times = energy_data_representation[energy_data_representation.time.dt.hour == 0].time
    weather_times = weather_data_representation[weather_data_representation.time.dt.hour == 0].time
    start_time = max(energy_times[0], weather_times[0])
    end_time = min(energy_times[-1], weather_times[-1])

    forecasting_time_points = pd.date_range(
        pd.to_datetime(start_time.values),
        pd.to_datetime(end_time.values),
        freq=f'{hparams.forecast_freq}h'
    )

    return {
        'energy_input': energy_data_representation.loc[forecasting_time_points],
        'weather_input': weather_data_representation.loc[forecasting_time_points]
    }


def create_preprocessing_pipeline(hparams):
    pipeline = Pipeline(path=os.path.join('run', 'preprocessing'))

    ###
    # Sanity checks of data
    ##
    # check and clean nan values
    valid_energy_data = FunctionModule(validate, name='ValidationEnergy')(input=pipeline['energy'])
    valid_weather_data = FunctionModule(validate, name='ValidationWeather')(input=pipeline['weather'])

    energy_preprocessing_pipeline, energy_normalizer = create_energy_preprocessing_pipeline(hparams)
    energy_preprocessing  = energy_preprocessing_pipeline(energy=valid_energy_data)
    weather_preprocessing_pipeline = create_weather_preprocessing_pipeline(hparams)
    weather_preprocessing = weather_preprocessing_pipeline(energy=valid_energy_data, weather=valid_weather_data)

    ###
    # Input and output generation
    ##
    # generate input and output targets for energy and weather
    dr_filtered = FunctionModule(
        lambda energy_data_representation, weather_data_representation: filter_data_representations(
            hparams, energy_data_representation, weather_data_representation
        ), name='DataRepresentationFilter'
    )(
        energy_data_representation=energy_preprocessing['EnergyDataRepresentationNormalized'],
        weather_data_representation=weather_preprocessing['WeatherDataRepresentationNormalized']
    )
    target_preprocessing_pipeline = create_target_preprocessing_pipeline(hparams, energy_normalizer)
    target_preprocessing = target_preprocessing_pipeline(
        energy_orig=valid_energy_data,
        energy_dr=dr_filtered['energy_input'])

    FunctionModule(lambda input, output: dr.calendar_features(hparams, input, output),
                   name='calendar_input')(
        input=dr_filtered['energy_input'], output=target_preprocessing['target_normalized']
    )
    FunctionModule(lambda input: input,
                   name='energy_input')(
        input=dr_filtered['energy_input']
    )
    FunctionModule(lambda input: input,
                   name='weather_input')(
        input=dr_filtered['weather_input']
    )
    FunctionModule(lambda input: input,
                   name='target_normalized')(
        input=target_preprocessing['target_normalized']
    )

    return pipeline, energy_normalizer