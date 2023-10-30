import numpy as np
from scipy.stats import distributions
from .variables import RandomVariable


def joint_normal(y, theta:list) -> np.ndarray:
    mu = theta[0]
    sigma = theta[1]
    log_p = distributions.norm.logpdf(y, loc=mu, scale=sigma)
    return np.sum(log_p)


def generate_proposal(random_variable:RandomVariable|list[RandomVariable], step_size=0.2) -> RandomVariable | list[RandomVariable]:
    random_variable = random_variable

    if isinstance(random_variable, list):
        proposed = []
        for r in random_variable:
            proposed.append(r.current + step_size * distributions.norm.rvs())
    else:
        proposed = random_variable.current + step_size * distributions.norm.rvs()
    
    return proposed


def mh_step(y:np.ndarray, thetas:list[RandomVariable], joint, step_size=0.5):
    # generate proposals for each theta
    thetas_current = []
    thetas_proposed = []

    for theta in thetas:
        thetas_current.append(theta.current)

    thetas_proposed = generate_proposal(thetas)
    
    # proposal log likelihood
    log_likelihood_proposed = joint_normal(y, thetas_proposed)
    log_prior_proposed = []

    for i in range(len(thetas)):
        log_prior_proposed.append(thetas[i].prior_likelihood(thetas_proposed[i]))
        
    log_prior_proposed = np.sum(log_prior_proposed)

    # current log likelihood
    log_likelihood_current = joint_normal(y, thetas_current)
    log_prior_current = []

    for i in range(len(thetas_current)):
        log_prior_current.append(thetas[i].prior_likelihood(thetas_current[i]))

    log_prior_current = np.sum(log_prior_current)

    log_r = log_likelihood_proposed - log_prior_proposed - log_likelihood_current + log_prior_current
    r = np.exp(log_r)
    
    p = np.min([1, r])

    u = distributions.uniform.rvs()

    if p > u:
        for i in range(len(thetas)):
            thetas[i].current = thetas_proposed[i]

    return thetas
