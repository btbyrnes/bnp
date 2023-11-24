from abc import ABC
import numpy as np

MH_SCALE_DEFAULT = 0.2


class RandomVariable(ABC):
    _constant:bool
    _current:float
    def __init__(self, current, constant=False) -> None: pass

    def new(self): ...

    def get_value(self) -> float: ...

    def set_value(self, current) -> None: 
        self._current = current

    def generate_rw_proposal(self, scale=MH_SCALE_DEFAULT) -> float: ...

    def generate_proposal(self): ...

    def log_prior(self, proposed) -> float: return 0.0

    def random_draw(self) -> np.floating | np.integer : ...

    def __repr__(self) -> str: ...

