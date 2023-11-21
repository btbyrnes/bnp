import numpy as np

from classes.variables import RandomVariable
from classes.dataset import DPMDataset

from .variables import RandomVariable, Normal, InvGamma
from .likelihood import Likelihood, NormalLikelihood
from .sampler import MHSampler
from .model import Model
# from .sampler import joint_normal, mh_step

LOG_ZERO = -1e4
P_ZERO = 1e-4

# Base Measure which is a group of random variables
# needs to be able to create variables for sampling
# and assign the current value to each respective
# parameter that it comes from
class BaseMeasure:
    _params:list[RandomVariable]
    def __init__(self, params:list[RandomVariable]) -> None:
        self._params = params

    def random_draw(self) -> list[np.floating | np.integer]:
        random = []
        for p in self._params:
            random.append(p.random_draw())
        return random
    
    def get_current(self) -> list[float]:
        current = []
        for p in self._params:
            current.append(p._current)
        return current
    
    def get_instance_with_values(self, values:list[float]) -> list[RandomVariable]:
        return_list = []
        for i, v in enumerate(values):
            # print("35 i: ", v)
            # print("36", self._params[i].new(current=v))
            return_list.append(self._params[i].new(current=v))

        return return_list
            
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"[" + ", ".join([str(p._current) for p in self._params]) + "]"
    

class NormalBaseMeasure(BaseMeasure):
    def __init__(self, params: list[RandomVariable]=[Normal(mu=0,sigma=3), InvGamma()]) -> None:
        super().__init__(params)
    

# Need to save each iteration of S and 
# the respective Phi for each S

class DPMChain:
    s: list[int]
    phi: list
    w: list
    n: list
    def __init__(self) -> None:
        self.s      = []
        self.phi    = []
        self.w      = []
        self.n      = []
    def __str__(self) -> str:
        return str(self.s) + "\n" + str(self.phi) + "\n" + str(self.w) + "\n" + str(self.n)


class DPM:
    _base_measure:BaseMeasure
    _likelihood:Likelihood
    _M:float
    _m:int                  = 1
    _measures:list          = []
    _weights:list[float]    = []
    _dataset:DPMDataset     = DPMDataset()
    _dpm_chain              = DPMChain()
    def __init__(self, 
                 base_measure:RandomVariable=NormalBaseMeasure(), 
                 likeilihood:Likelihood=NormalLikelihood(), 
                 data:DPMDataset=DPMDataset(),
                 M=1.0) -> None:
        self._base_measure = base_measure
        self._likelihood = likeilihood
        self._M = M

    def set_dataset(self, y:list|np.ndarray, s:list|np.ndarray) -> None:
        self._dataset.set_y(y)
        self._dataset.set_s(s)

    def set_measures(self, measures:list):
        self._measures = measures

    def get_measure(self, id:int) -> list:
        return self._measures[id]
    
    def set_weights(self):
        self._weights = self._dataset.calculate_weights()

    def draw_random_measure(self) -> BaseMeasure:
        return self._base_measure.random_draw()

    def get_new_clusters_to_sample(self, last_cluster:int, number_of_new:int) -> list:
        new_clusters = []
        for i in range(1, number_of_new + 1):
            new_clusters.append(last_cluster + 1)
        return new_clusters

    def get_new_measures_to_sample(self, number_of_new:int) -> list:
        new_measures = []
        for i in range(number_of_new):
            new_measures.append(self._base_measure.random_draw())
        return new_measures

    def pop_measures_by_cluster(self, clusters:list):
        measures = []
        for i in clusters:
            measures.append(self._measures[i])
        self._measures = measures

    def reset_clusters_to_zero(self, array_1:np.ndarray):
        # print("----")
        # print(array_1, id(array_1))
        array_2 = np.empty_like(array_1)
        # print(array_2, id(array_2))
        unique_in_1 = np.unique(array_1)
        # print(unique_in_1, id(unique_in_1))
        arg_sorted = unique_in_1.argsort()
        # print(arg_sorted, id(arg_sorted))


        for j in arg_sorted:
            array_2[np.where(array_1 == unique_in_1[j])] = arg_sorted[j]

        # print(array_2, id(array_2))
        # print("----")
        return array_2   
    
    def new_cluster_probability(self, M, m, n_total) -> float:
        return ((M / m) / (n_total + M))


    def sample(self, samples=10) -> None:
        for jj in range(samples):
            if jj % 10 == 0 and jj > 0: 
                print(f"step: {jj:>4d}/{jj:>4d} : {self._dataset.get_s()}")
            self.sample_over_cluster_assignments()
            self.mh_steps_over_each_cluster()
        # print("")
        # print(f"s: {self._dataset.get_s()}")


    def sample_over_cluster_assignments(self):
        DEBUG = False

        if DEBUG == True:
            print("---- ---- ---- ---- ---- ---- ---- ")
            print("  sample over cluster assignments  ")
            print("---- ---- ---- ---- ---- ---- ---- ")

        data = self._dataset

        m = self._m
        M = self._M

        y_ = data.get_y()
        # print(f"s: {data.get_s()}")
        for i in range(0, len(data)): # for each row
            if DEBUG == True:
                print(" ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ")
                print(f"{i:>2}:")
            ## Get the current clusters and their counts

            s = data.get_s()
            [current_clusters, n_j] = data.count_clusters()

            index_this_cluster = s[i]

            n_j[index_this_cluster] -= 1
            n_this_cluster = n_j[index_this_cluster]

            n_total = sum(n_j)
            current_measures = self._measures
            
            #### #### New candidate clusters
            ## Randomly draw 'm' new clusters
            last_cluster = max(current_clusters)

            new_measures = self.get_new_measures_to_sample(m)
            new_clusters = self.get_new_clusters_to_sample(last_cluster, m)


            current_prior_p = np.array(n_j) / (n_total + M)
            current_prior_p = list(current_prior_p)

            new_cluster_p = [self.new_cluster_probability(M, n_total, m) for _ in range(m)]

            proposed_measures = current_measures + new_measures
            proposed_clusters = current_clusters + new_clusters
            proposed_prior_p = current_prior_p + new_cluster_p

            
            # print(fcurrent_clusters, current_prior_p, len(current_clusters), len(current_prior_p))
            # if DEBUG == True:
            #     print("Current clusters:")
            #     # self.print_clusters_and_measures(current_clusters, current_measures)
            #     print("Proposed clusters:")
            #     # self.print_clusters_and_measures(proposed_clusters, proposed_measures)

            # This is probably fine because the take the most recent current value 
            # this is updated after each MH step
            # log of the measures
            y = y_[i]
            log_p_measures = self.cluster_likelihood(y, proposed_clusters, proposed_measures)

            # log of the prior_p due to cluster size
            log_p_sizes = self.cluster_size_probability(proposed_clusters, proposed_prior_p)
            # probability from log probability

            p = self.log_p_to_p(np.array(log_p_measures) + np.array(log_p_sizes))
            
            # if DEBUG == True:
            #     print(f"y:                 {y}")
            #     print(log_p_measures)
            #     print(n_j)
            #     print(log_p_sizes)
            #     print(f"p:                 {p}")

            chosen_cluster = self.random_choice(proposed_clusters, p)
           
            # if DEBUG == True:
            #     print(f"previous cluster:  {s[i]}")
            #     print(f"chosen cluster:    {chosen_cluster}")
            #     print(f"s: {s}, {id(s)}")
            s[i] = chosen_cluster

            # print(f"s: {s}, {id(s)}")

            new_clusters = []

            if chosen_cluster not in current_clusters:
                new_clusters = current_clusters + [chosen_cluster]
                self._measures = self._measures + [proposed_measures[chosen_cluster]]
            else:
                new_clusters = current_clusters

            if n_this_cluster == 0:
                self._measures.pop(index_this_cluster)
                new_clusters.pop(index_this_cluster)

            # Maybe need to call this outside the loop?
            s = self.reset_clusters_to_zero(s)
            data.set_s(s)
            if DEBUG == True:
                print(s, np.min(s), np.max(s), id(s))
                print(f"Done y_{i:>2d}")


        self.set_weights()
        n_unique = len(np.unique(data.get_s()))
        self._dpm_chain.w.append(self._weights)
        self._dpm_chain.s.append(list(s))
        self._dpm_chain.n.append(n_unique)


    def cluster_likelihood(self, y, clusters:list, measures:list) -> list:
        log_p = []
        for k in range(len(clusters)):
            params = measures[k]
            # print("235", params, type(params))
            # for p in params:
                # print("    237", p, type(p))
            lp = self._likelihood.log_likelihood(y, params)
            log_p.append(lp)
        return log_p


    def cluster_size_probability(self, clusters:list, cluster_p:list) -> list:
        log_p = []
        for l in range(len(clusters)):
            if cluster_p[l] > 0:
                log_p_this_cluster = np.log(cluster_p[l])
            else:
                log_p_this_cluster = LOG_ZERO
            log_p.append(log_p_this_cluster)
        return log_p


    def mh_steps_over_each_cluster(self, step=0.2):
        DEBUG = False
        if DEBUG == True:
            print("\n")
            print("---- ---- ---- ---- ---- ---- ---- ")
            print("    mh steps over each cluster     ")
            print("---- ---- ---- ---- ---- ---- ---- ")

        ## Instantiate a sampler

        ## Run the sampler

        ## Calculate the mean of the variable of the measure

        ## Set the mean for each parameter of the measure to the current for each measure
        y_ = self._dataset.get_y()
        s_ = self._dataset.get_s()
        # if DEBUG == True:
        # print(f"s: {s_}, {id(s_)}")

        likelihood = self._likelihood
        measures = self._measures

        number_of_clusters = len(measures)
        unique_clusters = np.unique(s_)

        # print(number_of_clusters, len(unique_clusters), unique_clusters)
        sampled_parameters = []
        # Assume clusters start from zero
        assert np.min(unique_clusters) == 0

        for i in range(number_of_clusters):
            # print(f"MH step {i}")
            # print(i, self._base_measure)
            
            y = np.array(y_)[np.where(np.array(s_) == i)[0]]

            # instantiate a model with the pre-defined likelihood, the
            # base measure, with current parameters by the current
            # cluster parameters

            params = self._base_measure.get_instance_with_values(measures[i])
            # print("268", params, type(params))

            # for p in params:
                # print("271", p, type(p))

            model = Model(params, likelihood)

            sampler = MHSampler(y, model)
            chain = sampler.sample()
            # print(chain.mean())
            # print(chain.std())
            sampled_parameters.append(chain.mean())
            # print("\n")

        # print(sampled_parameters)
        # print("s: ", s_, np.min(s_), np.max(s_), id(s_))     
        self._dpm_chain.phi.append(sampled_parameters)
        # print("Done MH step")


    # def print_clusters_and_measures(self, clusters, measures):
    #     for c, m in zip(clusters, measures):
    #         print(f"cluster {c:>2d} - {m}")
    #     print("")

    def random_choice(self, c:list, p:list) -> int:
        return np.random.choice(c, p=p)

    def log_p_to_p(self, log_p:list) -> list:
        p = np.exp(log_p - np.max(log_p))
        for ii in range(len(p)):
            if p[ii] <= P_ZERO: p[ii] = 0
        p = p / np.sum(p)

        return list(p)




# class DPM:
#     base_measure:list[RandomVariable]
#     M:float=1.0
#     RandomVariables:list
    
#     def  __init__(self, base_measure:list[RandomVariable], joint_probability=joint_normal, M=1.0) -> None:
#         if isinstance(base_measure, list):
#             self.base_measure = base_measure
#         else:
#             self.base_measure = [base_measure]

#         self.M = M
#         self.RandomVariables = []

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
        
#     def set_RandomVariables(self, RandomVariables:list[RandomVariable] | np.ndarray):
#         cols = len(self.base_measure)
#         if isinstance(RandomVariables, np.ndarray):
#             updated_params = []
#             rows = len(RandomVariables)
#             for i in range(rows):
#                 updated_params_row = []
#                 for j in range(cols):
#                     new_RandomVariable = self.base_measure[j].new(current=RandomVariables[i][j])
#                     updated_params_row.append(new_RandomVariable)
#                 updated_params.append(updated_params_row)
#             self.RandomVariables = updated_params
#         else:
#             self.RandomVariables = RandomVariables
#         logging.debug(f"RandomVariables: {self.RandomVariables}")

#     def set_weights(self, weights):
#         self.weights = weights

#     @property
#     def values(self) -> float | np.ndarray:
#         rows = len(self.RandomVariables)
#         cols = len(self.RandomVariables[0])
#         return_RandomVariables = np.zeros((rows, cols))

#         for i in range(rows):
#             for j in range(cols):
#                 return_RandomVariables[i][j] = self.RandomVariables[i][j].current
        
#         return return_RandomVariables


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

#     dpm.set_RandomVariables(thetas)
#     return s


# def sample_over_measures(y:np.ndarray, s:np.ndarray, dpm:DPM, likelihood=None, prior=None, step=0.2):
#     logger = logging.getLogger(__name__)
#     logging.debug(__name__)
#     thetas = dpm.RandomVariables
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

#         dpm.set_RandomVariables(thetas)

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