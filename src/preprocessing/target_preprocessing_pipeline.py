import os

import numpy as np
import xarray as xr

from pywatts.core.pipeline import Pipeline
from pywatts.modules.wrappers import FunctionModule
from src.pywatts import Normalizer


def filter_target(hparams, energy_orig, energy_dr):
    _, gt_idx, _ = np.intersect1d(energy_orig.time.values,
                                  energy_dr.time.values,
                                  return_indices=True)
    gt_idx = np.array([idx + np.arange(hparams.prediction_horizon) + 1 for idx in gt_idx])
    gt_values = energy_orig.values[gt_idx]
    return xr.DataArray(
        data=gt_values,
        coords={
            'time': energy_dr.time,
            'steps': np.arange(1, hparams.prediction_horizon + 1)
        },
        dims=['time', 'steps']
    )


def create_target_preprocessing_pipeline(hparams, energy_normalizer):
    pipeline = Pipeline(path=os.path.join('run', 'preprocessing', 'output'))

    model_output = FunctionModule(
        lambda energy_orig, energy_dr:
            filter_target(hparams, energy_orig, energy_dr),
        name='model_output'
    )(energy_orig=pipeline['energy_orig'], energy_dr=pipeline['energy_dr'])
    target_normalized = energy_normalizer(x=model_output)
    target_normalized = FunctionModule(lambda x: x, name='target_normalized')(
        x=target_normalized
    )

    return pipeline