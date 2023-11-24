import numpy as np

from classes.variables import RandomVariable
from classes.dataset import DPMDataset

from .variables import RandomVariable, Normal, InvGamma, Exponential
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
            return_list.append(self._params[i].new(current=v))

        return return_list
            
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"[" + ", ".join([str(p._current) for p in self._params]) + "]"
    

class NormalBaseMeasure(BaseMeasure):
    def __init__(self, params: list[RandomVariable]=[Normal(mu=0,sigma=3), InvGamma()]) -> None:
        super().__init__(params)
    
    
class ExponentialBaseMeasure(BaseMeasure):
    def __init__(self, params: list[RandomVariable]=[Exponential(4)]) -> None:
        super().__init__(params)


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
        array_2 = np.empty_like(array_1)
        unique_in_1 = np.unique(array_1)
        arg_sorted = unique_in_1.argsort()

        for j in arg_sorted:
            array_2[np.where(array_1 == unique_in_1[j])] = arg_sorted[j]

        return array_2   
    
    def new_cluster_probability(self, M, m, n_total) -> float:
        return ((M / m) / (n_total + M))


    def sample(self, samples=10) -> None:
        for jj in range(samples):
            if jj % int(samples/5) == 0 and jj > 0: 
                print(f"step: {jj:>4d}/{samples:>4d} : {self._dataset.get_s()}")
            self.sample_over_cluster_assignments()
            self.mh_steps_over_each_cluster()

    
    def sample_over_cluster_assignments(self):
        DEBUG = False

        data = self._dataset

        m = self._m
        M = self._M

        y_ = data.get_y()
 
        for i in range(0, len(data)): # for each row
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

            # This is probably fine because the take the most recent current value 
            # this is updated after each MH step
            # log of the measures
            y = y_[i]
            log_p_measures = self.cluster_likelihood(y, proposed_clusters, proposed_measures)

            # log of the prior_p due to cluster size
            log_p_sizes = self.cluster_size_probability(proposed_clusters, proposed_prior_p)
            # probability from log probability

            p = self.log_p_to_p(np.array(log_p_measures) + np.array(log_p_sizes))

            chosen_cluster = self.random_choice(proposed_clusters, p)
           
            s[i] = chosen_cluster

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

        self.set_weights()
        n_unique = len(np.unique(data.get_s()))
        self._dpm_chain.w.append(self._weights)
        self._dpm_chain.s.append(list(s))
        self._dpm_chain.n.append(n_unique)


    def cluster_likelihood(self, y, clusters:list, measures:list) -> list:
        log_p = []
        for k in range(len(clusters)):
            params = measures[k]
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

        # Instantiate a sampler
        # Run the sampler
        # Calculate the mean of the variable of the measure
        # Set the mean for each parameter of the measure to the current for each measure
        y_ = self._dataset.get_y()
        s_ = self._dataset.get_s()

        likelihood = self._likelihood
        measures = self._measures

        number_of_clusters = len(measures)
        unique_clusters = np.unique(s_)

        sampled_parameters = []
        
        # Assume clusters start from zero
        assert np.min(unique_clusters) == 0

        for i in range(number_of_clusters):
            y = np.array(y_)[np.where(np.array(s_) == i)[0]]

            # instantiate a model with the pre-defined likelihood, the
            # base measure, with current parameters by the current
            # cluster parameters

            params = self._base_measure.get_instance_with_values(measures[i])
            model = Model(params, likelihood)

            sampler = MHSampler(y, model)
            chain = sampler.rw_sample(scale=0.2)
            sampled_parameters.append(chain.mean())

        self._dpm_chain.phi.append(sampled_parameters)
    
    def random_choice(self, c:list, p:list) -> int:
        return np.random.choice(c, p=p)

    def log_p_to_p(self, log_p:list) -> list:
        p = np.exp(log_p - np.max(log_p))
        for ii in range(len(p)):
            if p[ii] <= P_ZERO: p[ii] = 0
        p = p / np.sum(p)

        return list(p)