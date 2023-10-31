import os

import numpy as np

from pywatts.core.pipeline import Pipeline
from pywatts.modules import Sampler, Slicer
from pywatts.modules.wrappers import FunctionModule

import src.data_representations as dr
from src.pywatts import Normalizer


def create_energy_preprocessing_pipeline(hparams):
    pipeline = Pipeline(path=os.path.join('run', 'preprocessing', 'energy'))

    # energy data normalization
    energy_normalizer = Normalizer(method=hparams.scaling, name='EnergyNormalizer')
    normalized_energy_data = energy_normalizer(x=pipeline['energy'])

    # calculate energy lag features (time, energy_lag_features)
    # WARNING: all energy data representations are based on that
    energy_lag_features = Sampler(sample_size=hparams.energy_lag_features, name='EnergyLagFeatures')(x=normalized_energy_data)
    energy_lag_features = Slicer(hparams.energy_lag_features - 1, name='EnergyFeaturesSliced')(x=energy_lag_features)

    # energy data representations
    if f'energy_{hparams.energy_data_representation_type}_fit' in dr.__dict__:
        energy_data_representation = FunctionModule(
                lambda input: getattr(dr, f'energy_{hparams.energy_data_representation_type}')(hparams, input),
                lambda input: getattr(dr, f'energy_{hparams.energy_data_representation_type}_fit')(hparams, input),
                name='EnergyDataRepresentation'
            )(input=energy_lag_features)
    else:
        energy_data_representation = FunctionModule(
                lambda input: getattr(dr, f'energy_{hparams.energy_data_representation_type}')(hparams, input),
                name='EnergyDataRepresentation'
            )(input=energy_lag_features)

    # energy data normalization
    energy_dr_normalizer = Normalizer(method=hparams.scaling, name='EnergyDataRepresentationNormalized')
    normalized_energy_dr = energy_dr_normalizer(x=energy_data_representation)

    return pipeline, energy_normalizer