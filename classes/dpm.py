import numpy as np
from .variables import Likelihood, Normal, Parameter
# from .sampler import joint_normal, mh_step

import logging

class Measure:
    _params:list[Parameter]
    _likelihood:Likelihood
    def __init__(self, params:list[Parameter]) -> None:
        self._params = params

    def random_sample(self) -> list[Parameter]:
        random = []
        for p in self._params:
            random.append(p.random_draw())
        return 



class DPM:
    _base_measure:Variable  = Normal()
    _M:float                = 1.0
    _measures:list          = [Variable]
    _m:int                  = 1

    def __init__(self, base_measure:Variable=Normal(), M=1.0) -> None:
        self._base_measure = base_measure
        self._M = M

    def add_measure(self, measure:Variable):
        self._measures.append(measure)


    def sample_over_cluster_assignments(self, y_:np.ndarray, s_:np.ndarray):
        m = self._m

        for i in range(y_): # for each row
            y = np.delete(y_, i)
            s = np.delete(s_, i)

            c_j, n_j = np.unique(s, return_counts=True)
            last_c_j = np.max(c_j)
            h = last_c_j + m









# def sample_cluster_assignemnts(y:np.ndarray, s:np.ndarray, dpm:DPM, m=1):
#     thetas = dpm.values
#     M = dpm.M
#     for i in range(len(y)):
#         c_j, n_j = np.unique(s, return_counts=True)
#         # Ensure we start with clusters indexed from zero
#         assert np.min(c_j) == 0

#         data_mask = np.ones(s.shape, dtype=bool)
#         data_mask[i] = 0

#         c_i = s[i]
        
#         last_c_j = np.max(c_j)
#         h = last_c_j + m

#         theta_h = np.array([dpm.sample_measure()])

#         clusters_to_sample = np.append(c_j, np.array(range(last_c_j + 1, h + 1)) )
#         thetas_to_sample = np.concatenate((thetas, theta_h))

#         N = np.sum(n_j)
#         n_j[c_i] = n_j[c_i] - 1

#         sample_coefficients = n_j / (N - 1 + M)
#         new_cluster_coefficient = (M / m) / (N - 1 + M)

#         sample_coefficients = np.append(sample_coefficients, np.ones(m) * new_cluster_coefficient)

#         # Check the sizes of clusters and the sample coefficients
#         assert clusters_to_sample.shape == sample_coefficients.shape

#         cluster_theta_mask = np.ones_like(clusters_to_sample, dtype=bool)

#         if n_j[c_i] < 1:
#             cluster_theta_mask[c_i] = False
#         else:
#             pass

#         clusters_to_sample_masked = clusters_to_sample[cluster_theta_mask]

#         log_p = dpm.log_p(y[i], thetas_to_sample=thetas_to_sample, sample_coefficients=sample_coefficients) # + np.log(sample_coefficients)
#         log_p = log_p[clusters_to_sample_masked]

#         p = log_p_to_p(log_p)
#         p = p / np.sum(p)
 
#         new_cluster = np.random.choice(clusters_to_sample_masked, p=p)
#         if (s[i] != new_cluster):
#             logging.debug(f"{i:>2}: Assiging new cluster {s[i]} to {new_cluster}")
#         s[i] = new_cluster

#         c_j, n_j = np.unique(s, return_counts=True)
#         thetas = thetas_to_sample[c_j]

#         # Now reset the clusters
#         s = reset_clusters_index_to_zero(s)

#     dpm.set_parameters(thetas)
#     return s





# class DPM:
#     base_measure:list[RandomVariable]
#     M:float=1.0
#     parameters:list
    
#     def  __init__(self, base_measure:list[RandomVariable], joint_probability=joint_normal, M=1.0) -> None:
#         if isinstance(base_measure, list):
#             self.base_measure = base_measure
#         else:
#             self.base_measure = [base_measure]

#         self.M = M
#         self.parameters = []

#     def sample_measure(self) -> list:
#         random_measures = []
#         for i in range(len(self.base_measure)):
#             random_measures.append(self.base_measure[i].random_sample())

#         return random_measures

#     def log_p(self, y, thetas_to_sample, sample_coefficients) -> np.ndarray:
#         log_p = np.zeros(shape=len(thetas_to_sample))

#         for i in range(len(thetas_to_sample)):
#             log_p[i] = joint_normal(y, theta=thetas_to_sample[i]) + np.log(sample_coefficients[i])
        
        
#         return log_p
        
#     def set_parameters(self, parameters:list[RandomVariable] | np.ndarray):
#         cols = len(self.base_measure)
#         if isinstance(parameters, np.ndarray):
#             updated_params = []
#             rows = len(parameters)
#             for i in range(rows):
#                 updated_params_row = []
#                 for j in range(cols):
#                     new_parameter = self.base_measure[j].new(current=parameters[i][j])
#                     updated_params_row.append(new_parameter)
#                 updated_params.append(updated_params_row)
#             self.parameters = updated_params
#         else:
#             self.parameters = parameters
#         logging.debug(f"parameters: {self.parameters}")

#     def set_weights(self, weights):
#         self.weights = weights

#     @property
#     def values(self) -> float | np.ndarray:
#         rows = len(self.parameters)
#         cols = len(self.parameters[0])
#         return_parameters = np.zeros((rows, cols))

#         for i in range(rows):
#             for j in range(cols):
#                 return_parameters[i][j] = self.parameters[i][j].current
        
#         return return_parameters


# def sample_cluster_assignemnts(y:np.ndarray, s:np.ndarray, dpm:DPM, m=1):
#     thetas = dpm.values
#     M = dpm.M
#     for i in range(len(y)):
#         c_j, n_j = np.unique(s, return_counts=True)
#         # Ensure we start with clusters indexed from zero
#         assert np.min(c_j) == 0

#         data_mask = np.ones(s.shape, dtype=bool)
#         data_mask[i] = 0

#         c_i = s[i]
        
#         last_c_j = np.max(c_j)
#         h = last_c_j + m

#         theta_h = np.array([dpm.sample_measure()])

#         clusters_to_sample = np.append(c_j, np.array(range(last_c_j + 1, h + 1)) )
#         thetas_to_sample = np.concatenate((thetas, theta_h))

#         N = np.sum(n_j)
#         n_j[c_i] = n_j[c_i] - 1

#         sample_coefficients = n_j / (N - 1 + M)
#         new_cluster_coefficient = (M / m) / (N - 1 + M)

#         sample_coefficients = np.append(sample_coefficients, np.ones(m) * new_cluster_coefficient)

#         # Check the sizes of clusters and the sample coefficients
#         assert clusters_to_sample.shape == sample_coefficients.shape

#         cluster_theta_mask = np.ones_like(clusters_to_sample, dtype=bool)

#         if n_j[c_i] < 1:
#             cluster_theta_mask[c_i] = False
#         else:
#             pass

#         clusters_to_sample_masked = clusters_to_sample[cluster_theta_mask]

#         log_p = dpm.log_p(y[i], thetas_to_sample=thetas_to_sample, sample_coefficients=sample_coefficients) # + np.log(sample_coefficients)
#         log_p = log_p[clusters_to_sample_masked]

#         p = log_p_to_p(log_p)
#         p = p / np.sum(p)
 
#         new_cluster = np.random.choice(clusters_to_sample_masked, p=p)
#         if (s[i] != new_cluster):
#             logging.debug(f"{i:>2}: Assiging new cluster {s[i]} to {new_cluster}")
#         s[i] = new_cluster

#         c_j, n_j = np.unique(s, return_counts=True)
#         thetas = thetas_to_sample[c_j]

#         # Now reset the clusters
#         s = reset_clusters_index_to_zero(s)

#     dpm.set_parameters(thetas)
#     return s


# def sample_over_measures(y:np.ndarray, s:np.ndarray, dpm:DPM, likelihood=None, prior=None, step=0.2):
#     logger = logging.getLogger(__name__)
#     logging.debug(__name__)
#     thetas = dpm.parameters
#     rows = len(thetas)
#     results = []
#     logging.debug(f"thetas: {thetas} {type(thetas)}")
#     logging.debug(f"rows:   {rows}")

#     c_j, n_j = np.unique(s, return_counts=True)

#     logging.debug(f"c_j:    {c_j} {len(c_j)}")

#     assert len(c_j) == rows

#     for row in range(rows):
#         theta = thetas[row]
#         y_ = y[s == c_j[row]]
#         logging.debug(f"y_: {y_}")
#         logging.debug(f"c_j: {c_j}")
#         result, _ = mh_step(y, thetas=theta, likelihood=joint_normal, step_size=step)
#         results.append(result)

#     logging.debug(f"results: {results}")
#     return results



# def dpm_sampler_7(x:np.ndarray, s:np.ndarray, dpm:DPM, burn_in:int=100, lag:int=5, mh_steps:int=1):
    
#     results = {"c": [], "n": [], "w": [], "n_clusters": [], "theta": []}

#     for i in range(1000):
#         s = sample_cluster_assignemnts(x, s, dpm)
        
#         for j in range(mh_steps):
#             thetas = sample_over_measures(x, s, dpm, step=.2)

#         dpm.set_parameters(thetas)

#         sample_p = dpm.values

#         sample_p = np.array(sample_p)
#         w = cluster_weights(s)
#         n = len(np.unique(s))

#         if i > burn_in and i % lag == 0:
#             results["w"].append(w)
#             results["theta"].append(sample_p)
#             results["n_clusters"].append(n)

#     return results



# def log_p_to_p(log_p:np.ndarray) -> np.ndarray:
#     return np.exp(log_p - np.max(log_p))

# def cluster_means(clusters:np.ndarray, y:np.ndarray) -> np.ndarray:
#     r = clusters

#     uniques, counts = np.unique(r, return_counts=True)

#     means = np.zeros_like(uniques)

#     for i, u in enumerate(uniques):
#         y_u = y[r == u]
#         mean = np.mean(y_u)
#         means[i] = mean

#     return means

# def cluster_weights(clusters:np.ndarray) -> np.ndarray:
#     r = clusters

#     uniques, counts = np.unique(r, return_counts=True)

#     return counts / np.sum(counts)

# # def init_clusters(y:np.ndarray, H=10):
# #     assert y.ndim == 1

# #     Z = hierarchy.linkage(y.reshape(-1,1))
# #     r = hierarchy.cut_tree(Z, n_clusters=10).squeeze()

# #     means = cluster_means(r, y)
# #     weights = cluster_weights(r, y)

# #     return means, weights, r, np.repeat(0, H)

# def reset_clusters_index_to_zero(array_1:np.ndarray) -> np.ndarray:
#     array_2 = np.empty_like(array_1)
#     unique_in_1 = np.unique(array_1)
#     arg_sorted = unique_in_1.argsort()

#     for j in arg_sorted:
#         array_2[np.where(array_1 == unique_in_1[j])] = arg_sorted[j]

#     return array_2   





# # def log_likelihood(y, theta):
# #     return np.sum(distributions.norm.logpdf(y, loc=theta))

# # def log_prior(theta, theta_prior=0.0):
# #     return distributions.norm.logpdf(theta, loc=theta_prior)

# # def mh_steps(y:np.ndarray, thetas:np.ndarray, likelihood=None, prior=None, step=0.2):
# #     rows, _ = thetas.shape

# #     results = np.zeros(shape=thetas.shape)
# #     for row in range(rows):
# #         mu, sigma = thetas[row, :]
# #         result = np.array([[mh_step(y, mu, likelihood=likelihood, prior=prior, step=step), sigma]])
# #         results[row] = result

# #     return results

# # # def mh_step(y, theta, likelihood=None, prior=None, step=0.2):
# # #     theta_proposed = distributions.norm.rvs(loc=theta, scale=step*1)
# # #     log_p = log_likelihood(y, theta_proposed) + log_prior(theta) - log_likelihood(y, theta) - log_prior(theta_proposed)

# # #     a = np.exp(log_p)
# # #     p = np.min([1, a])
# # #     u = distributions.uniform.rvs(loc=0, scale=1)

# # #     if p > u:
# # #         theta = theta_proposed

# # #     return theta  