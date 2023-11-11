from scipy.stats import distributions
from classes.variables.base import Parameter, MH_SCALE_DEFAULT


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

    def random_draw(self):
        a = self._shape
        scale = self._scale
        return distributions.invgamma.rvs(a=a, scale=scale)

    def __str__(self) -> str:
        return f"IG({self._shape}, {self._scale}) - const={self._constant} - current={self._current:.3f}"