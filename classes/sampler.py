import numpy as np
from scipy.stats import distributions
from classes.variables import Variable

import logging


def mh_step(y:np.ndarray, var:Variable):
    proposed = var.generate_mh_proposals()
    
    log_current = var.log_likelihood(y)
    log_proposed = proposed.log_likelihood(y)

    log_alpha = log_proposed - log_current
    alpha = np.exp(log_alpha)
    p = np.min([1, alpha])

    u = distributions.uniform.rvs()

    if p > u:
        return proposed
    else:
        return var
    

class MHSampler:
    _data:np.ndarray
    _variable:Variable
    _chain:list

    def __init__(self, y, variable:Variable) -> None:
        self._data = y
        self._variable = variable

    def sample(self, steps=100, burn_in=10, lag=5):
        chain = []
        y = self._data
        variable = self._variable
        new_var = variable
        for i in range(steps):
            new_var = mh_step(y, new_var)
            if i > burn_in and i % lag == 0:
                chain.append(new_var)
        self.set_chain(chain)

    def set_chain(self, chain):
        self._chain = chain


def get_chain_parameters(chain:list[Variable]):
    parameters = []
    for row in chain:
        parameters.append(row.get_values())

    return parameters


# def normal(y, theta:np.ndarray):
#     logger = logging.getLogger(__name__)
#     assert isinstance(theta, np.ndarray)

#     logger.info(f"theta: {theta} {type(theta)} {theta.ndim} {theta.shape}")
#     logger.info(f"y: {y}")
#     logger.info(f"mean y: {np.mean(y)}")
#     if theta.ndim != 2:
#         theta = np.array([theta])
#     mu = theta[:, 0]
#     sigma = 1
#     log_p = distributions.norm.logpdf(y, loc=mu, scale=sigma)
#     logger.info(f"log_p: {log_p}")
#     return np.sum(log_p)


# def joint_normal(y, theta:np.ndarray):    # Type output this
#     logger = logging.getLogger(__name__)
#     assert isinstance(theta, np.ndarray)

#     logger.info(f"theta: {theta} {type(theta)} {theta.ndim} {theta.shape}")
#     logger.info(f"y: {y}")
#     logger.info(f"mean y: {np.mean(y)}")
#     if theta.ndim != 2:
#         theta = np.array([theta])
#     mu = theta[:, 0]
#     sigma = theta[:, 1]
#     log_p = distributions.norm.logpdf(y, loc=mu, scale=sigma)
#     logger.info(f"log_p: {log_p}")
#     return np.sum(log_p)


# def generate_symmetric_proposal(random_variable:list[RandomVariable]|RandomVariable, step_size) -> np.ndarray:
#     logger = logging.getLogger(__name__)
#     random_variable = random_variable

#     if isinstance(random_variable, list):
#         # logging.debug(f"if isinstance(random_variable, list): {isinstance(random_variable, list)}")
#         proposed = []
#         for r in random_variable:
#             proposed.append(r.current + step_size * distributions.norm.rvs())
#     else:
#         proposed = random_variable.current + step_size * distributions.norm.rvs()
    
#     # logger.debug(f"proposed: {proposed}")
#     return np.array(proposed)


# def mh_step(y:np.ndarray, thetas:list[RandomVariable], likelihood, step_size):
#     logger = logging.getLogger(__name__)
#     logger.info(__name__)
#     # generate proposals for each theta
#     thetas_current = []
#     thetas_proposed = []

#     # logger.debug(f"theta: {thetas}")
#     for t in thetas:
#         thetas_current.append(t.value)
#     thetas_current = np.array(thetas_current)
#     thetas_proposed = generate_symmetric_proposal(thetas, step_size=step_size)

#     logging.info(f"thetas_current:  {thetas_current}") 
#     logging.info(f"thetas_proposed: {thetas_proposed}") 
#     # logger.debug(f"theta_current:  {thetas_current} {thetas_current.shape}")
#     # logger.debug(f"theta_proposed: {thetas_proposed} {thetas_proposed.shape}")

#     # proposal log likelihood
#     log_likelihood_proposed = joint_normal(y, thetas_proposed)

#     log_prior_proposed = []
#     log_prior_current = []

#     for i in range(len(thetas)):
#         log_prior_proposed.append(thetas[i].prior_likelihood(thetas_proposed[i]))
#     log_prior_proposed = np.sum(log_prior_proposed)

#     # current log likelihood
#     log_likelihood_current = joint_normal(y, thetas_current)


#     for i in range(len(thetas_current)):
#         log_prior_current.append(thetas[i].prior_likelihood(thetas_current[i]))
#     log_prior_current = np.sum(log_prior_current)

#     logger.info(f"log_likelihood_proposed: {log_likelihood_proposed}")
#     logger.info(f"log_likelihood_current:  {log_likelihood_current}")
#     logger.info(f"log_prior_current:       {log_prior_current}")
#     logger.info(f"log_prior_proposed:      {log_prior_proposed}")
#     log_r = log_likelihood_proposed + log_prior_proposed - log_likelihood_current - log_prior_current
#     logger.info(f"log_r:                   {log_r}")
#     r = np.exp(log_r)
#     logger.info(f"r: {r}")
    
#     p = np.min([1, r])

#     u = distributions.uniform.rvs()
#     accept = 0
#     if p > u:
#         accept = 1
#         logger.debug(f"p > u: {p} > {u}")
#         for i in range(len(thetas)):
#             thetas[i].current = thetas_proposed[i]
#         # logging.debug(f"thetas: {thetas}")
#     return thetas, accept
