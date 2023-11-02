from abc import ABC
import numpy as np
from scipy.stats import distributions


class RandomVariable(ABC):
    prior: ...
    current: ...
    proposed: ...

    def __init__(self) -> None: ...

    def prior_likelihood(self) -> float: ...

    def random_sample(self) -> float: ...

    @property
    def value(self) -> float:
        return self.current

    def new(self, current): ...


class Normal(RandomVariable):
    def __init__(self, mu=0.0, sigma=1.0, current=0.0) -> None:
        self.prior = [mu, sigma]
        self.current = current

    def prior_likelihood(self, y:float) -> float:
        mu, sigma = self.prior
        return distributions.norm.logpdf(y, loc=mu, scale=sigma)
    
    def random_sample(self) -> float:
        mu, sigma = self.prior
        return distributions.norm.rvs(loc=mu, scale=sigma)

    def new(self, current:float):
        mu, sigma = self.prior
        return Normal(mu=mu, sigma=sigma, current=current)

    def __repr__(self) -> str:
        mu, sigma = self.prior
        return f"Normal({mu},{sigma}^2) - current: {self.current}"
    

class Gamma(RandomVariable):
    def __init__(self, shape=1.0, scale=2.0, current=1.0) -> None:
        self.prior = [shape, scale]
        self.current = current

    def prior_likelihood(self, y:float) -> float:
        shape, scale = self.prior
        return distributions.gamma.logpdf(y, a=shape, scale=scale)
    
    def random_sample(self) -> float:
        shape, scale = self.prior
        return distributions.gamma.rvs(a=shape, scale=scale)
    
    def new(self, current:float):
        shape, scale = self.prior
        return Gamma(shape=shape, scale=scale, current=current)

    def __repr__(self) -> str:
        shape, scale = self.prior
        return f"Gamma({shape},{scale}) - current: {self.current}"