import numpy as np
from scipy.stats import distributions
from classes.variables import RandomVariable, Normal, Gamma


def gibbs_step_normal_likelihood(y, thetas:list[RandomVariable]):
    assert isinstance(thetas[0], Normal) and isinstance(thetas[1], Gamma)

    thetas_current = []
    for t in thetas:
        thetas_current.append(t.value)
    thetas_current = np.array(thetas_current)

    n = len(y)

    prior_mu, prior_sigma = thetas[0].prior

    y_sum = np.sum(y)
    y_bar = np.mean(y)

    alpha_prior, scale_prior = thetas[1].prior


    alpha = alpha_prior + n/2
    beta = (1/scale_prior) + (0.5)*np.sum((y - y_bar)**2)

    tau_proposed = distributions.gamma.rvs(a=alpha, scale=1/beta)

    mu = tau_proposed / (n*tau_proposed + 1/prior_sigma) * y_sum
    sigma = 1 / (n*tau_proposed + 1/prior_sigma)

    mu_proposed = distributions.norm.rvs(loc=mu, scale=sigma)

    return mu_proposed, tau_proposed