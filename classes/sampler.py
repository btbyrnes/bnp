import numpy as np
from scipy.stats import distributions
from classes.likelihood import Likelihood
from classes.variables import Parameter
from classes.model import Model
import logging

MH_SCALE_DEFAULT = 0.2

class MHSampler:
    _data:np.ndarray
    _model:Model
    _chain:list
    _acceptances:int    = 0
    _samples:int        = 0

    def __init__(self, y, model:Model) -> None:
        self._data = y
        self._model = model

    def sample(self, steps=100, burn_in=10, lag=5):
        chain = []
        y = self._data
        for i in range(steps):
            self.mh_step(y)
            if i > burn_in and i % lag == 0:
                chain.append(self._model.current_parameters())

        self.set_chain(chain)
        print(f"Acceptance Rate: {self._acceptances}/{self._samples} = {100*self._acceptances/float(self._samples):.3f}%")

    def mh_step(self, y:np.ndarray, scale=MH_SCALE_DEFAULT) -> list[Parameter]:
        likelihood = self._model._likelihood
        current = self._model._parameters
        proposal = []

        for p in current:
            proposed = p.generate_mh_proposal(scale)
            proposal.append(proposed)

        log_current = likelihood.log_likelihood(y, current)
        log_proposed = likelihood.log_likelihood(y, proposal)

        log_alpha = log_proposed - log_current
        alpha = np.exp(log_alpha)
        p = np.min([1, alpha])

        u = distributions.uniform.rvs()

        self._samples += 1
        if p > u:
            self._acceptances += 1
            self._model._parameters = proposal

    def set_chain(self, chain):
        self._chain = chain

    def get_chain(self):
        chain = self._chain
        parameters = []
        for row in chain:
            parameters.append(row)

        return parameters