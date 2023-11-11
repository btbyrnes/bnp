import numpy as np
from scipy.stats import distributions
from classes.variables.base import Parameter, Variable, MH_SCALE_DEFAULT
from classes.variables.exponential import ExponentialParameter


class Poisson(Variable):
    _params:list[Parameter] = [ExponentialParameter()]
    def __init__(self, params:list[Parameter]=None):
        if params: super().__init__(params)
    
    def log_likelihood(self, y:np.ndarray) -> float:
        mu = self._params[0].get_value()

        return np.sum(distributions.poisson.logpmf(k=y, mu=mu))
    
    def generate_mh_proposals(self, scale=MH_SCALE_DEFAULT) -> Variable:
        proposals = []
        params = self._params

        for p in params:
            proposed = p.generate_mh_proposal(scale=scale)
            proposals.append(proposed)

        y = Poisson(proposals)

        return y
            
    def __getitem__(self, key):
        assert key < len(self._params)
        return self._params[key]

    def __str__(self) -> str:
        return f"Poisson({self._params[0]} - current={self._params[0]._current:.3f}"