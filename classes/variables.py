from abc import ABC
import numpy as np
from scipy.stats import distributions


MH_SCALE_DEFAULT = 0.2

class Parameter(ABC):
    _constant:bool
    _current:float
    def __init__(self, current, constant=False) -> None: pass

    def get_value(self) -> float: ...

    def set_value(self) -> None: ...

    def generate_mh_proposal(self, scale=MH_SCALE_DEFAULT):
        # if self._constant == True: return self._current
        # else: raise Exception("Parameter's ABC can only return constant values")
        pass

    def __repr__(self) -> str: return str(self)


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

    def __str__(self) -> str:
        return f"N({self._mu}, {self._sigma}^2) - constant={self._constant} - current={self._current:.3f}"


class InvGammaParameter(Parameter):
    _shape:float    = 1.0
    _scale:float    = 1.0
    _current:float  = 1.0
    _constant:bool  = False

    def __init__(self, shape=0.0, scale=1.0, current=1.0, constant=False) -> None:
        self._shape = shape
        self._scale = scale
        self._current = current
        self._constant = constant

    def get_value(self) -> float:
        return self._current
    
    def set_current(self, current:float) -> float:
        try:
            assert current >= 0.0
            self._current = current
        except:
            raise Exception("Attempted to set InvGamma to negative, outside of support")
    
    def generate_mh_proposal(self, scale=MH_SCALE_DEFAULT) -> Parameter:
        if (self._constant == True): return InvGammaParameter(self._shape, self._scale, self._current, self._constant)
        else:
            proposed = self._current + distributions.norm.rvs(scale=scale)
            if proposed <= 0.0: proposed = 1e-5
            y = InvGammaParameter(self._shape, self._scale, proposed, self._constant)
            return y
        
    def __str__(self) -> str:
        return f"IG({self._shape}, {self._scale}) - constant={self._constant} - current={self._current:.3f}"


class Variable(ABC):
    _params: ...
    def __init__(self, params:list[Parameter]): self._params = params

    def get_values(self) -> float: return [p.get_value() for p in self._params]

    def generate_mh_proposals(self): ...

    def log_likelihood(self, y) -> float: ...

    def __repr__(self) -> str: return str(self)


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


def mh_step(y:np.ndarray, var:Normal):
    proposed = var.generate_mh_proposals()
    
    log_current = var.log_likelihood(y)
    log_proposed = proposed.log_likelihood(y)

    log_alpha = log_proposed - log_current
    alpha = np.exp(log_alpha)
    p = np.min([1, alpha])

    u = distributions.uniform.rvs()

    if p > u:
        print("accept")
        return proposed
    else:
        print("reject")
        return var



# class RandomVariable(ABC):
#     prior: ...
#     current: ...
#     proposed: ...

#     def __init__(self) -> None: ...

#     def prior_likelihood(self) -> float: ...

#     def random_sample(self) -> float: ...

#     @property
#     def value(self) -> float:
#         return self.current

#     def new(self, current): ...


# class Normal(RandomVariable):
#     def __init__(self, mu=0.0, sigma=1.0, current=0.0) -> None:
#         self.prior = [mu, sigma]
#         self.current = current

#     def prior_likelihood(self, y:float) -> float:
#         mu, sigma = self.prior
#         return distributions.norm.logpdf(y, loc=mu, scale=sigma)
    
#     def random_sample(self) -> float:
#         mu, sigma = self.prior
#         return distributions.norm.rvs(loc=mu, scale=sigma)

#     def new(self, current:float):
#         mu, sigma = self.prior
#         return Normal(mu=mu, sigma=sigma, current=current)

#     def __repr__(self) -> str:
#         mu, sigma = self.prior
#         return f"Normal({mu},{sigma}^2) - current: {self.current}"
    

# class Gamma(RandomVariable):
#     def __init__(self, shape=1.0, scale=2.0, current=1.0) -> None:
#         self.prior = [shape, scale]
#         self.current = current

#     def prior_likelihood(self, y:float) -> float:
#         shape, scale = self.prior
#         return distributions.gamma.logpdf(y, a=shape, scale=scale)
    
#     def random_sample(self) -> float:
#         shape, scale = self.prior
#         return distributions.gamma.rvs(a=shape, scale=scale)
    
#     def new(self, current:float):
#         shape, scale = self.prior
#         return Gamma(shape=shape, scale=scale, current=current)

#     def __repr__(self) -> str:
#         shape, scale = self.prior
#         return f"Gamma({shape},{scale}) - current: {self.current}"
    

# class InvGamma(RandomVariable):
#     def __init__(self, shape=1.0, scale=1.0, current=1.0) -> None:
#         self.prior = [shape, scale]
#         self.current = current

#     def prior_likelihood(self, y:float) -> float:
#         shape, scale = self.prior
#         return distributions.invgamma.logpdf(y, a=shape, scale=scale)
    
#     def random_sample(self) -> float:
#         shape, scale = self.prior
#         return distributions.invgamma.rvs(a=shape, scale=scale)
    
#     def new(self, current:float):
#         shape, scale = self.prior
#         return InvGamma(shape=shape, scale=scale, current=current)

#     def __repr__(self) -> str:
#         shape, scale = self.prior
#         return f"InvGamma({shape},{scale}) - current: {self.current}"

