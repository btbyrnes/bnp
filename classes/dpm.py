import numpy as np
from variables import RandomVariable


class DPM:
    base_measure:list[RandomVariable]
    M:float=1.0

    def  __init__(self, base_measure:list[RandomVariable], M=1.0) -> None:
        if isinstance(base_measure, list):
            self.base_measure = base_measure
        else:
            self.base_measure = [base_measure]

        self.M = M

        self.current_measures = []


    def initialize_measures(self):
        pass

    def sample_measure(self):
        pass

    def log_p(self):
        pass

    def set_weights(self, weights):
        self.weights = weights

    def set_measures(self, measures:list):
        self.current_measures = measures


def dpm_sampler():
    pass

def sample_cluster_assignemnts(y:np.ndarray, s:np.ndarray, dpm:DPM, m=1):
    thetas = dpm.measures
    M = dpm.M
    for i in range(len(y)):
        c_j, n_j = np.unique(s, return_counts=True)
        # Ensure we start with clusters indexed from zero
        assert np.min(c_j) == 0

        data_mask = np.ones(s.shape, dtype=bool)
        data_mask[i] = 0

        c_i = s[i]
        
        last_c_j = np.max(c_j)
        h = last_c_j + m

        theta_h = [dpm.random_sample()]

        clusters_to_sample = np.append(c_j, np.array(range(last_c_j + 1, h + 1)) )
        thetas_to_sample = np.concatenate((thetas, theta_h))

        N = np.sum(n_j)
        n_j[c_i] = n_j[c_i] - 1
        
        sample_coefficients = n_j / (N - 1 + M)
        new_cluster_coefficient = (M / m) / (N - 1 + M)

        sample_coefficients = np.append(sample_coefficients, np.ones(m) * new_cluster_coefficient)

        cluster_theta_mask = np.ones_like(clusters_to_sample, dtype=bool)

        if n_j[c_i] < 1:
            cluster_theta_mask[c_i] = False
        else:
            pass

        clusters_to_sample_masked = clusters_to_sample[cluster_theta_mask]

        log_p = dpm.log_p(y[i], theta=thetas_to_sample)  # + np.log(sample_coefficients)
        log_p = log_p[clusters_to_sample_masked]

        p = log_p_to_p(log_p)
        p = p / np.sum(p)

        new_cluster = np.random.choice(clusters_to_sample_masked, p=p)

        s[i] = new_cluster

        c_j, n_j = np.unique(s, return_counts=True)
        thetas = thetas_to_sample[c_j]

        # Now reset the clusters
        s = reset_clusters_index_to_zero(s)

    dpm.measures = thetas

    return s

def log_likelihood(y, theta):
    return np.sum(distributions.norm.logpdf(y, loc=theta))

def log_prior(theta, theta_prior=0.0):
    return distributions.norm.logpdf(theta, loc=theta_prior)

def mh_steps(y:np.ndarray, thetas:np.ndarray, likelihood=None, prior=None, step=0.2):
    rows, _ = thetas.shape

    results = np.zeros(shape=thetas.shape)
    for row in range(rows):
        mu, sigma = thetas[row, :]
        result = np.array([[mh_step(y, mu, likelihood=likelihood, prior=prior, step=step), sigma]])
        results[row] = result

    return results

def mh_step(y, theta, likelihood=None, prior=None, step=0.2):
    theta_proposed = distributions.norm.rvs(loc=theta, scale=step*1)
    log_p = log_likelihood(y, theta_proposed) + log_prior(theta) - log_likelihood(y, theta) - log_prior(theta_proposed)

    a = np.exp(log_p)
    p = np.min([1, a])
    u = distributions.uniform.rvs(loc=0, scale=1)

    if p > u:
        theta = theta_proposed

    return theta  

def log_p_to_p(log_p:np.ndarray) -> np.ndarray:
    return np.exp(log_p - np.max(log_p))

def cluster_means(clusters:np.ndarray, y:np.ndarray) -> np.ndarray:
    r = clusters

    uniques, counts = np.unique(r, return_counts=True)

    means = np.zeros_like(uniques)

    for i, u in enumerate(uniques):
        y_u = y[r == u]
        mean = np.mean(y_u)
        means[i] = mean

    return means

def cluster_weights(clusters:np.ndarray) -> np.ndarray:
    r = clusters

    uniques, counts = np.unique(r, return_counts=True)

    return counts / np.sum(counts)

def init_clusters(y:np.ndarray, H=10):
    assert y.ndim == 1

    Z = hierarchy.linkage(y.reshape(-1,1))
    r = hierarchy.cut_tree(Z, n_clusters=10).squeeze()

    means = cluster_means(r, y)
    weights = cluster_weights(r, y)

    return means, weights, r, np.repeat(0, H)

def reset_clusters_index_to_zero(array_1:np.ndarray) -> np.ndarray:
    array_2 = np.empty_like(array_1)
    unique_in_1 = np.unique(array_1)
    arg_sorted = unique_in_1.argsort()

    for j in arg_sorted:
        array_2[np.where(array_1 == unique_in_1[j])] = arg_sorted[j]

    return array_2   
