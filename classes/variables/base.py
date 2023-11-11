from abc import ABC


MH_SCALE_DEFAULT = 0.2


class Parameter(ABC):
    _constant:bool
    _current:float
    def __init__(self, current, constant=False) -> None: pass

    def get_value(self) -> float: ...

    def set_value(self) -> None: ...

    def generate_mh_proposal(self, scale=MH_SCALE_DEFAULT): ...

    def random_draw(self): ...

    def __repr__(self) -> str: ...


class Variable(ABC):
    _params: ...
    def __init__(self, params:list[Parameter]): self._params = params

    def get_values(self) -> float: return [p.get_value() for p in self._params]

    def generate_mh_proposals(self): ...

    def log_likelihood(self, y) -> float: ...

    def __repr__(self) -> str: return str(self)
