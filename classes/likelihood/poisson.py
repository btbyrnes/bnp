import numpy as np
from scipy.stats import distributions
from classes.variables import RandomVariable, Exponential
from classes.variables.base import MH_SCALE_DEFAULT

from classes.likelihood.base import Likelihood

class PoissonLikelihood(Likelihood):
    def __init__(self, params:list[RandomVariable]=None):
        if params: super().__init__(params)
    
    @classmethod
    def log_likelihood(cls, y:np.ndarray, params:list[RandomVariable]) -> float:
        mu = params[0].get_value()
        return np.sum(distributions.poisson.logpmf(k=y, mu=mu))
    
    # def generate_mh_proposals(self, scale=MH_SCALE_DEFAULT) -> Likelihood:
    #     proposals = []
    #     params = self._params

    #     for p in params:
    #         proposed = p.generate_mh_proposal(scale=scale)
    #         proposals.append(proposed)

    #     y = PoissonLikelihood(proposals)

    #     return y
            
    # def __getitem__(self, key):
    #     assert key < len(self._params)
    #     return self._params[key]

    def __str__(self) -> str:
        return f"Poisson likelihood"