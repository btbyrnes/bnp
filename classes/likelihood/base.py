from abc import ABC
from classes.variables.base import RandomVariable


class Likelihood(ABC):
    # _params:list[RandomVariable] = []
    # def __init__(self, params:list[RandomVariable]): self._params = params

    # def get_values(self) -> float: return [p.get_value() for p in self._params]

    # def set_parameters(self, update:list[float]):
    #     try: 
    #         assert len(update) == len(self._params)
    #         for i in range(len(self._params)):
    #             self._params[i].set_value(update[i])
    #     except:
    #         raise Exception("Wrong number of update values for given parameters")

    # def generate_mh_proposals(self) -> list[RandomVariable]: ...
    
    @classmethod
    def log_likelihood(self, y, parmeters:list[RandomVariable]) -> float: ...

    def __repr__(self) -> str: return str(self)
