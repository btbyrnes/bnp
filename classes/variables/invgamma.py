from scipy.stats import distributions
from classes.variables.base import RandomVariable, MH_SCALE_DEFAULT


class InvGamma(RandomVariable):
    _shape:float    = 1.0
    _scale:float    = 1.0
    _current:float  = 1.0
    _constant:bool  = False

    def __init__(self, shape=1.0, scale=1.0, current=1.0, constant=False) -> None:
        self._shape = shape
        self._scale = scale
        self._current = current
        self._constant = constant

    def new(self, current=None):
        if current is None: current = self._current
        x = InvGamma(self._shape, self._scale, current, self._constant)
        # print("20", x, type(x), id(x))
        return x

    def get_value(self) -> float:
        return self._current
    
    def get_current(self) -> float:
        return self._current

    def set_current(self, current:float) -> float:
        try:
            assert current >= 0.0
            self._current = current
        except:
            raise Exception("Attempted to set InvGamma to negative, outside of support")
    
    def generate_mh_proposal(self, scale=MH_SCALE_DEFAULT) -> RandomVariable:
        if (self._constant == True): return InvGamma(self._shape, self._scale, self._current, self._constant)
        else:
            proposed = self._current + distributions.norm.rvs(scale=scale)
            if proposed <= 0.0: proposed = 1e-5
            y = InvGamma(self._shape, self._scale, proposed, self._constant)
            return y

    def random_draw(self):
        a = self._shape
        scale = self._scale
        return distributions.invgamma.rvs(a=a, scale=scale)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"IG({self._shape:.3f}, {self._scale:.3f}) = {self._current:.3f}"