import numpy as np
from scipy.stats import distributions
from .variables import RandomVariable

import logging

def joint_normal(y, theta:np.ndarray):    # Type output this
    logger = logging.getLogger(__name__)
    assert isinstance(theta, np.ndarray)

    logger.info(f"theta: {theta} {type(theta)} {theta.ndim} {theta.shape}")
    if theta.ndim != 2:
        theta = np.array([theta])
    mu = theta[:, 0]
    sigma = theta[:, 1]
    log_p = distributions.norm.logpdf(y, loc=mu, scale=sigma)

    return log_p


def generate_proposal(random_variable:list[RandomVariable]|RandomVariable, step_size=0.2) -> np.ndarray:
    logger = logging.getLogger(__name__)
    random_variable = random_variable

    if isinstance(random_variable, list):
        # logging.info(f"if isinstance(random_variable, list): {isinstance(random_variable, list)}")
        proposed = []
        for r in random_variable:
            proposed.append(r.current + step_size * distributions.norm.rvs())
    else:
        proposed = random_variable.current + step_size * distributions.norm.rvs()
    
    # logger.info(f"proposed: {proposed}")
    return np.array(proposed)


def mh_step(y:np.ndarray, thetas:list[RandomVariable], likelihood, step_size=0.5):
    logger = logging.getLogger(__name__)
    # generate proposals for each theta
    thetas_current = []
    thetas_proposed = []

    # logger.info(f"theta: {thetas}")
    for t in thetas:
        thetas_current.append(t.value)
    thetas_current = np.array(thetas_current)
    thetas_proposed = generate_proposal(thetas)    
    # logger.info(f"theta_current:  {thetas_current} {thetas_current.shape}")
    # logger.info(f"theta_proposed: {thetas_proposed} {thetas_proposed.shape}")

    # proposal log likelihood
    log_likelihood_proposed = joint_normal(y, thetas_proposed)
    log_likelihood_proposed = np.sum(log_likelihood_proposed)

    log_prior_proposed = []
    for i in range(len(thetas)):
        log_prior_proposed.append(thetas[i].prior_likelihood(thetas_proposed[i]))
    log_prior_proposed = np.sum(log_prior_proposed)

    # current log likelihood
    log_likelihood_current = joint_normal(y, thetas_current)
    log_likelihood_current = np.sum(log_likelihood_current)

    # logging.info(f"log_likelihood_proposed: {log_likelihood_proposed}")
    # logging.info(f"log_likelihood_current:  {log_likelihood_current}")

    log_prior_current = []

    for i in range(len(thetas_current)):
        log_prior_current.append(thetas[i].prior_likelihood(thetas_current[i]))
    log_prior_current = np.sum(log_prior_current)

    log_r = log_likelihood_proposed - log_prior_proposed - log_likelihood_current + log_prior_current
    r = np.exp(log_r)

    logger.info(f"r: {r}")
    
    p = np.min([1, r])

    u = distributions.uniform.rvs()

    if p > u:
        logger.info(f"p > u: {p} > {u}")
        for i in range(len(thetas)):
            thetas[i].current = thetas_proposed[i]
        # logging.info(f"thetas: {thetas}")
    return thetas
