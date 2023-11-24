import numpy as np
from scipy.stats import distributions
from classes.likelihood import Likelihood
from classes.variables import RandomVariable
from classes.model import Model
import logging

MH_SCALE_DEFAULT = 0.2
UNDERFLOW_LOG_MIN = -20

class Chain:
    _variables:list         = []
    _chain_contents:list    = []
    def __init__(self, variables:list[RandomVariable]):
        self._variables = variables
        chain_contents = []
        for v in variables:
            chain_contents.append([])
        self._chain_contents = chain_contents

    def __getitem__(self, id):
        try:
            assert id < len(self._chain_contents)
        except:
            raise IndexError(f"Asked for {id}, chain parameters are {len(self._chain_contents)}")
        return self._chain_contents[id]
    
    def __len__(self):
        return len(self._chain_contents)

    def add_obseration(self, params:list[RandomVariable]) -> None:
        for i, p in enumerate(params):
            current_param = self._chain_contents[i]
            current_param.append(p.get_value())
    
    def get_chain(self) -> list:
        return self._chain_contents
    
    def mean(self) -> list[float]:
        chain_mean = []
        for i in range(len(self._chain_contents)):
            mean = np.mean(self._chain_contents[i])
            chain_mean.append(mean)
        return chain_mean
    
    def std(self) -> list[float]:
        chain_std = []
        for i in range(len(self._chain_contents)):
            mean = np.std(self._chain_contents[i])
            chain_std.append(mean)
        return chain_std


class MHSampler:
    _data:np.ndarray
    _model:Model
    _chain:Chain
    _acceptances:int    = 0
    _samples:int        = 0

    def __init__(self, y, model:Model) -> None:
        self._data = y
        self._model = model
        self._chain = Chain(model._parameters)

    def rw_sample(self, steps=100, burn_in=10, lag=5, scale=MH_SCALE_DEFAULT) -> Chain:
        y = self._data
        for i in range(steps):
            self.mh_rw_step(y, scale)
            if i > burn_in and i % lag == 0:
                current_params = self._model.current_parameters()
                self._chain.add_obseration(current_params)

        # print(f"Acceptance Rate: {self._acceptances}/{self._samples} = {100*self._acceptances/float(self._samples):.3f}%")
        return self._chain
    
    def sample(self, steps=100, burn_in=10, lag=5) -> Chain:
        y = self._data
        for i in range(steps):
            self.mh_independent_step(y)
            if i > burn_in and i % lag == 0:
                current_params = self._model.current_parameters()
                self._chain.add_obseration(current_params)
        # print(f"Acceptance Rate: {self._acceptances}/{self._samples} = {100*self._acceptances/float(self._samples):.3f}%")
        return self._chain

    def mh_rw_step(self, y:np.ndarray, scale=MH_SCALE_DEFAULT) -> list[RandomVariable]:
        likelihood = self._model._likelihood
        current = self._model._parameters
        proposal = []

        for p in current:
            proposed = p.generate_rw_proposal(scale)
            proposal.append(proposed)

        log_prior_current = []
        for i, p in enumerate(current):
            lp = p.log_prior(y=p.get_value())
            log_prior_current.append(lp)

        log_prior_proposed = []
        for i, p in enumerate(proposal):
            lc = current[i].log_prior(y=p)
            log_prior_proposed.append(lc)

        log_current = likelihood.log_likelihood(y, current) + np.sum(log_prior_current)
        log_proposed = likelihood.log_likelihood(y, proposal) + np.sum(log_prior_proposed)

        log_alpha = log_proposed - log_current

        if log_alpha < UNDERFLOW_LOG_MIN:
            log_alpha = UNDERFLOW_LOG_MIN

        alpha = np.exp(log_alpha)
        p = np.min([1, alpha])

        u = distributions.uniform.rvs()

        self._samples += 1
        if p > u:
            self._acceptances += 1
            self._model.update_parameters(proposal)

    def update_parameters(self, update):
        self._model.update_parameters(update)

    def parameter_means(self) -> list[float]:
        chain = self._chain
        means = []
        for c in chain._chain_contents:
            means.append(np.mean(c))
        return means

    def get_chain(self):
        chain = self._chain
        return chain
    
    def mh_independent_step(self, y:np.ndarray) -> list[RandomVariable]:
        likelihood = self._model._likelihood
        current = self._model._parameters
        proposal = []

        for p in current:
            proposed = p.generate_proposal()
            proposal.append(proposed)

        log_prior_current = []
        for i, p in enumerate(current):
            lp = p.log_prior(y=p.get_value())
            log_prior_current.append(lp)

        log_prior_proposed = []
        for i, p in enumerate(proposal):
            lc = current[i].log_prior(y=p)
            log_prior_proposed.append(lc)

        log_current = likelihood.log_likelihood(y, current) + sum(log_prior_current)
        log_proposed = likelihood.log_likelihood(y, proposal) + sum(log_prior_proposed)

        log_alpha = log_proposed - log_current

        if log_alpha < UNDERFLOW_LOG_MIN:
            log_alpha = UNDERFLOW_LOG_MIN
        
        alpha = np.exp(log_alpha)

        p = np.min([1, alpha])

        u = distributions.uniform.rvs()

        self._samples += 1
        if p > u:
            self._acceptances += 1
            self._model.update_parameters(proposal)