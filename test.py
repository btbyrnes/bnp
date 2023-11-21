import numpy as np
from classes.dpm import BaseMeasure, DPM, NormalBaseMeasure
from classes.likelihood import NormalLikelihood
from classes.variables import Normal, InvGamma

if __name__ == "__main__":
   y = np.array([-1.42064144,  0.68615248, -0.12268064, -1.56841365,  0.77278432,
                 -2.9957832 , -0.80084822, -1.5276294 , -0.96945657, -0.42762472,
                 0.06589666, 1.72620247, 2.53837067, 2.06008666])
   s = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 2, 2])

   measures = [
      [-5, 1],
      [-2, 1],
      [2, 1],
      [5, 1]]

   base_measure = NormalBaseMeasure([Normal(mu=0,sigma=3), InvGamma()])
   dpm = DPM(base_measure)

   dpm.set_dataset(y, s)

   dpm.set_measures(measures)
   dpm.set_weights()

   # dpm.sample_over_cluster_assignments()

   # s = dpm._dataset.get_s()
   # dpm.mh_steps_over_each_cluster(y, s)

   dpm.sample(3)






