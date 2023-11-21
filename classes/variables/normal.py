import numpy as np
from scipy.stats import distributions
from classes.variables.base import RandomVariable, MH_SCALE_DEFAULT


class Normal(RandomVariable):
    _mu:float       = 0.0
    _sigma:float    = 1.0
    _current:float  = 0.0
    _constant:float = False
    def __init__(self, mu=0.0, sigma=1.0, current=0.0, constant=False) -> None:
        self._mu = mu
        self._sigma = sigma
        self._current = current
        self._constant = constant

    def new(self, current=None):
        # Get a copy of this measure with the priors specified and 
        # some new value
        if current is None: current = self._current
        x = Normal(self._mu, self._sigma, current, self._constant)
        # print("22", x, type(x), id(x))
        return x

    def set_current(self, current:float) -> float:
        self._current = current

    def get_value(self) -> float:
        return self._current
    
    def get_current(self) -> float:
        return self._current
    
    def generate_mh_proposal(self, scale=MH_SCALE_DEFAULT) -> RandomVariable:
        if (self._constant == True): return Normal(self._mu, self._sigma, self._current, self._constant)
        else:
            proposed = self._current + distributions.norm.rvs(scale=scale)
            y = Normal(self._mu, self._sigma, proposed, self._constant)
            return y

    def random_draw(self) -> np.float64:
        mu = self._mu
        sigma = self._sigma
        return distributions.norm.rvs(loc=mu, scale=sigma)
    
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"N({self._mu:.3f}, {self._sigma:.3f}^2) = {self._current:.3f}"

