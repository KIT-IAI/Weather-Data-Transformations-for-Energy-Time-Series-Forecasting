import os
import string
import random
from enum import Enum

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from pywatts.callbacks import PrintCallback
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import FunctionModule
from pywatts.modules.wrappers import FunctionModule, SKLearnWrapper
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from src.preprocessing import create_preprocessing_pipeline
from src.pywatts import ModelHandler

class Models(Enum):
    LINEAR = 0
    RF = 1
    SVR = 2
    DNN = 3


def replace_negative_forecasts(input):
    input.values[input < 0] = 0
    return input


def div_capacity(hparams, energy, capacity):
    cap = capacity.loc[energy.time].values
    result = energy.values.flatten() / cap.flatten()
    return numpy_to_xarray(result, energy)


def mul_capacity(hparams, energy, capacity):
    _, cap_idx, _ = np.intersect1d(capacity.time.values,
                                    energy.time.values,
                                    return_indices=True)
    cap_idx = np.array([idx + np.arange(energy.shape[1]) for idx in cap_idx])
    cap_values = capacity.values[cap_idx]
    result = energy * cap_values
    return numpy_to_xarray(result, energy)


class MySKLearnWrapper(SKLearnWrapper):

    def __init__(self, hparams, **kwargs):
        self.hparams = hparams
        super().__init__(**kwargs)

    def fit(self, energy, weather, calendar, target_y, **kwargs):
        print(self.module)

        time = energy.time
        split_time = pd.to_datetime(self.hparams.train_split)
        train_idx = time < split_time

        energy = energy.loc[train_idx]
        weather = weather.loc[train_idx]
        calendar = calendar.loc[train_idx]
        target_y = target_y.loc[train_idx]
        super().fit(energy=energy, weather=weather, calendar=calendar, target_y=target_y)


def create_pipeline(hparams):
    """
    Set up pywatts pipeline to preprocess, train, predict and postprocess
    data for the energy forecasting use case and make evaluations.
    """

    ###
    # Set up pipeline and callbacks for debugging
    ##
    pipeline = Pipeline(path=os.path.join('run', ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))))
    callbacks = [PrintCallback()] if hparams.debugging else []

    ###
    # Preprocessing pipeline
    # returning (calendar_input, energy_input, weather_input, target_normalized)
    ##
    energy = FunctionModule(
        lambda energy, capacity: div_capacity(hparams, energy, capacity),
        name='EnergyNormalizeCapacity'
    )(energy=pipeline['energy'], capacity=pipeline['energy_capacity'])
    preprocessing_pipe, target_normalizer = create_preprocessing_pipeline(hparams)
    preprocessing = preprocessing_pipe(
        energy=energy, weather=pipeline['weather'],
        callbacks=callbacks
    )

    ###
    # Model training
    ##
    if hparams.model == Models.LINEAR:
        forecast_normalized = MySKLearnWrapper(hparams, module=LinearRegression(n_jobs=-1), name='LinearModel')(
            energy=preprocessing['energy_input'], weather=preprocessing['weather_input'],
            calendar=preprocessing['calendar_input'], target_y=preprocessing['target_normalized'])
    if hparams.model == Models.RF:
        forecast_normalized = MySKLearnWrapper(hparams, module=RandomForestRegressor(n_jobs=-1, n_estimators=25, max_depth=7, min_samples_leaf=3), name='RandomForestRegressor')(
            energy=preprocessing['energy_input'], weather=preprocessing['weather_input'],
            calendar=preprocessing['calendar_input'], target_y=preprocessing['target_normalized'])
    if hparams.model == Models.SVR:
        forecast_normalized = MySKLearnWrapper(hparams, module=SVR(n_jobs=-1), name='SVR')(
            energy=preprocessing['energy_input'], weather=preprocessing['weather_input'],
            calendar=preprocessing['calendar_input'], target_y=preprocessing['target_normalized'])
    elif hparams.model == Models.DNN:
        forecast_normalized = ModelHandler(hparams=hparams, name='ModelHandler')(
            energy=preprocessing['energy_input'], weather=preprocessing['weather_input'],
            calendar=preprocessing['calendar_input'], y=preprocessing['target_normalized'])

    ###
    # Reverse normalization and return 'ground_truth' and 'forecast'
    ##
    forecast = target_normalizer(
        x=forecast_normalized,
        computation_mode=ComputationMode.Transform,
        use_inverse_transform=True
    )
    forecast = FunctionModule(
        lambda energy, capacity: mul_capacity(hparams, energy, capacity),
        name='ForecastInverteNormalizeCapacity'
    )(energy=forecast, capacity=pipeline['energy_capacity'])
    forecast = FunctionModule(replace_negative_forecasts, name='forecast')(input=forecast)
    ground_truth = target_normalizer(
        x=preprocessing['target_normalized'],
        computation_mode=ComputationMode.Transform,
        use_inverse_transform=True
    )
    ground_truth = FunctionModule(
        lambda energy, capacity: mul_capacity(hparams, energy, capacity),
        name='ground_truth'
    )(energy=ground_truth, capacity=pipeline['energy_capacity'])


    return pipeline
