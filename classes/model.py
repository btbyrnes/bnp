from .variables import RandomVariable
from .likelihood import Likelihood

class Model:
    _parameters:list[RandomVariable]
    _likelihood:Likelihood

    def __init__(self, parameters:list[RandomVariable], likeilihood:Likelihood) -> None:
        self._parameters = parameters
        self._likelihood = likeilihood

    def update_parameters(self, update:list[float] = None) -> None:
        try:
            assert len(update) == len(self._parameters)
            for i in range(len(self._parameters)):
                self._parameters[i].set_value(update[i])
        except:
            raise Exception("Wrong number of update parameters to model parameters")
        
    def current_parameters(self) -> list[RandomVariable]:
        return self._parameters


    
