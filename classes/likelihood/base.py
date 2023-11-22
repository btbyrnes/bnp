from abc import ABC
from classes.variables.base import RandomVariable


class Likelihood(ABC):    
    @classmethod
    def log_likelihood(self, y, parmeters:list[RandomVariable]) -> float: ...

    def __repr__(self) -> str: return str(self)
