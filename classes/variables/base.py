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

    def generate_mh_proposal(self, scale=MH_SCALE_DEFAULT): ...

    def log_prior(self, proposed) -> float: ...

    def random_draw(self) -> np.floating | np.integer : ...

    def __repr__(self) -> str: ...

