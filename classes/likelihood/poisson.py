import numpy as np
from scipy.stats import distributions
from classes.variables import RandomVariable, Exponential
from classes.variables.base import MH_SCALE_DEFAULT

from classes.likelihood.base import Likelihood

class PoissonLikelihood(Likelihood):
    def __init__(self, params:list[RandomVariable]=None):
        if params: super().__init__(params)
    
    @classmethod
    def log_likelihood(cls, y:np.ndarray, params:list) -> float:
        mu = params[0]
        if isinstance(mu, RandomVariable): mu = mu.get_value()
        return np.sum(distributions.poisson.logpmf(k=y, mu=mu))

    def __str__(self) -> str:
        return f"Poisson likelihood"