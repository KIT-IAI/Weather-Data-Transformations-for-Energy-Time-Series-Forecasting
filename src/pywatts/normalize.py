from typing import Dict

from pywatts.core.base import BaseEstimator


class Normalizer(BaseEstimator):

    def __init__(self, method: str = 'standard', name: str = 'Normalizer'):
        super().__init__(name)
        self.method = method.lower()
        self.has_inverse_transform = True
        self.stats = None

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def fit(self, x):
        if self.method == 'standard':
            self.stats = dict(
                mean=x.mean(),
                std=x.std()
            )
        elif self.method == 'minmax':
            self.stats = dict(
                min=x.min(),
                max=x.max()
            )
        elif self.method == 'none':
            pass
        else:
            raise ValueError(f'Unkown normalization method {self.method}.')
        self.is_fitted = True

    def transform(self, x):
        if self.method == 'standard':
            mean, std = self.stats['mean'], self.stats['std']
            x_scaled = (x - mean) / std
        elif self.method == 'minmax':
            min, max = self.stats['min'], self.stats['max']
            x_scaled = (x - min) / (max - min)
        elif self.method == 'none':
            x_scaled = x
        else:
            raise ValueError(f'Unkown normalization method {self.method}.')

        return x_scaled

    def inverse_transform(self, x):
        if self.method == 'standard':
            mean, std = self.stats['mean'], self.stats['std']
            x_inverse_scaled = x * std + mean
        elif self.method == 'minmax':
            min, max = self.stats['min'], self.stats['max']
            x_inverse_scaled = x * (max - min) + min
        elif self.method == 'none':
            x_inverse_scaled = x
        else:
            raise ValueError(f'Unkown normalization method {self.method}.')

        return x_inverse_scaled
