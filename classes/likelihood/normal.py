import numpy as np
from scipy.stats import distributions
from classes.variables import RandomVariable, Normal, InvGamma
from classes.variables.base import MH_SCALE_DEFAULT

from classes.likelihood.base import Likelihood


class NormalLikelihood(Likelihood):
    # def __init__(self, params:list[RandomVariable]=None):
    #     if params: super().__init__(params)

    @classmethod
    def log_likelihood(cls, y:np.ndarray, params:list) -> float:
        mu = params[0]
        if isinstance(mu, RandomVariable): mu=mu.get_value()
        sigma = params[1]
        if isinstance(sigma, RandomVariable): sigma=sigma.get_value()

        return np.sum(distributions.norm.logpdf(x=y, loc=mu, scale=sigma))
            
    # def __getitem__(self, key):
    #     assert key < len(self._params)
    #     return self._params[key]

    def __str__(self) -> str:
        # return f"Normal - current={self._params[0]._current:.3f},{self._params[1]._current:.3f}"
        return "Normal likelihood"