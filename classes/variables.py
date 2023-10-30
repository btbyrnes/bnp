from abc import ABC
import numpy as np
from scipy.stats import distributions


class RandomVariable(ABC):
    prior: ...
    current: ...
    proposed: ...

    def __init__(self) -> None:
        ...

    def prior_likelihood(self) -> float:
        ...


class Normal(RandomVariable):
    def __init__(self, mu=0.0, sigma=1.0, current=0.0) -> None:
        self.prior = [mu, sigma]
        self.current = current

    def prior_likelihood(self, y:float) -> float:
        mu, sigma = self.prior
        return distributions.norm.logpdf(y, loc=mu, scale=sigma)
    

class Gamma(RandomVariable):
    def __init__(self, shape=1.0, scale=2.0, current=1.0) -> None:
        self.prior = [shape, scale]
        self.current = current

    def prior_likelihood(self, y:float) -> float:
        shape, scale = self.prior
        return distributions.gamma.logpdf(y, a=shape, scale=scale)