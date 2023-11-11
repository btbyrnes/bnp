import numpy as np
from scipy.stats import distributions
from classes.variables.base import Parameter, Variable, MH_SCALE_DEFAULT
from classes.variables.invgamma import InvGammaParameter


class NormalParameter(Parameter):
    _mu:float       = 0.0
    _sigma:float    = 1.0
    _current:float  = 0.0
    _constant:float = False
    def __init__(self, mu=0.0, sigma=1.0, current=0.0, constant=False) -> None:
        self._mu = mu
        self._sigma = sigma
        self._current = current
        self._constant = constant

    def get_value(self) -> float:
        return self._current
    
    def set_value(self, current) -> None:
        self._current = current
    
    def generate_mh_proposal(self, scale=MH_SCALE_DEFAULT) -> Parameter:
        if (self._constant == True): return NormalParameter(self._mu, self._sigma, self._current, self._constant)
        else:
            proposed = self._current + distributions.norm.rvs(scale=scale)
            y = NormalParameter(self._mu, self._sigma, proposed, self._constant)
            return y

    def random_draw(self):
        mu = self._mu
        sigma = self._sigma
        return distributions.norm.rvs(loc=mu, scale=sigma)

    def __str__(self) -> str:
        return f"N({self._mu}, {self._sigma}^2) - const={self._constant} - current={self._current:.3f}"


class Normal(Variable):
    _params:list[Parameter] = [NormalParameter(), InvGammaParameter()]
    def __init__(self, params:list[Parameter]=None):
        if params: super().__init__(params)
    
    def log_likelihood(self, y:np.ndarray) -> float:
        mu = self._params[0].get_value()
        sigma = self._params[1].get_value()

        return np.sum(distributions.norm.logpdf(x=y, loc=mu, scale=sigma))
    
    def generate_mh_proposals(self, scale=MH_SCALE_DEFAULT) -> Variable:
        proposals = []
        params = self._params

        for p in params:
            proposed = p.generate_mh_proposal(scale=scale)
            proposals.append(proposed)

        y = Normal(proposals)

        return y
            
    def __getitem__(self, key):
        assert key < len(self._params)
        return self._params[key]

    def __str__(self) -> str:
        return f"N({self._params[0]}, {self._params[1]}^2) - current={self._params[0]._current:.3f},{self._params[1]._current:.3f}"