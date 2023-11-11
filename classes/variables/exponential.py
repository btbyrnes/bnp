from scipy.stats import distributions
from classes.variables import Parameter
from classes.variables.base import MH_SCALE_DEFAULT


class Exponential(Parameter):
    _loc:float      = 1.0
    _current:float  = 1.0
    _constant:bool  = False

    def __init__(self, loc=1.0, current=1.0, constant=False) -> None:
        self._shape = loc
        self._current = current
        self._constant = constant

    def get_value(self) -> float:
        return self._current
    
    def set_current(self, current:float) -> float:
        try:
            assert current >= 0.0
            self._current = current
        except:
            raise Exception("Attempted to set Exponential to negative, outside of support")
    
    def generate_mh_proposal(self, scale=MH_SCALE_DEFAULT) -> Parameter:
        if (self._constant == True): return Exponential(self._shape, self._current, self._constant)
        else:
            proposed = self._current + distributions.norm.rvs(scale=scale)
            if proposed <= 0.0: proposed = 1e-5
            y = Exponential(self._shape, proposed, self._constant)
            return y

    def random_draw(self):
        a = self._loc
        return distributions.expon.rvs(mu=a)

    def __str__(self) -> str:
        return f"Exp({self._loc} - const={self._constant} - current={self._current:.3f}"