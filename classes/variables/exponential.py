from scipy.stats import distributions
from classes.variables import RandomVariable
from classes.variables.base import MH_SCALE_DEFAULT


class Exponential(RandomVariable):
    _lam:float      = 1.0
    _current:float  = 1.0
    _constant:bool  = False

    def __init__(self, lam=1.0, current=1.0, constant=False) -> None:
        self._lam = lam
        self._current = current
        self._constant = constant
    
    def new(self, current=None):
        if current is None: current = self._current
        x = Exponential(self._lam, current, self._constant)
        return x

    def get_value(self) -> float:
        return self._current
    
    def set_current(self, current:float) -> float:
        try:
            assert current >= 0.0
            self._current = current
        except:
            raise Exception("Attempted to set Exponential to negative, outside of support")
    
    def generate_proposal(self, scale=MH_SCALE_DEFAULT) -> float:
        if (self._constant == True): return self.get_value()
        else:
            proposed = distributions.expon.rvs(scale=self._lam)
            assert proposed > 0
            return float(proposed)

    def generate_rw_proposal(self, scale=MH_SCALE_DEFAULT) -> float:
        if (self._constant == True): return Exponential(self._lam, self._current, self._constant)
        else:
            proposed = self._current + distributions.norm.rvs(scale=scale)
            if proposed <= 0.0: proposed = 1e-5
            # y = Exponential(self._lam, proposed, self._constant)
            return proposed       

    def log_prior(self, y:float) -> float:
        log_p = 0.0
        log_p = distributions.expon.logpdf(y, scale=self._lam)
        log_p = float(log_p)
        return log_p

    def random_draw(self):
        a = self._lam
        return distributions.expon.rvs(scale=a)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f"Exp({self._lam} - const={self._constant} - current={self._current:.3f}"