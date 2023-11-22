import numpy as np
from scipy.stats import distributions
from classes.variables import RandomVariable, Normal, InvGamma
from classes.variables.base import MH_SCALE_DEFAULT

from classes.likelihood.base import Likelihood


class NormalLikelihood(Likelihood):
    @classmethod
    def log_likelihood(cls, y:np.ndarray, params:list) -> float:
        mu = params[0]
        if isinstance(mu, RandomVariable): mu=mu.get_value()
        sigma = params[1]
        if isinstance(sigma, RandomVariable): sigma=sigma.get_value()

        return np.sum(distributions.norm.logpdf(x=y, loc=mu, scale=sigma))

    def __str__(self) -> str:
        return "Normal likelihood"