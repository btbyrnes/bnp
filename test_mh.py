import numpy as np
# from classes.dpm import Measure, DPM, NormalBaseMeasure
from classes.likelihood import NormalLikelihood
from classes.variables import Normal, InvGamma
from classes.sampler import MHSampler
from classes.model import Model

if __name__ == "__main__":
   y = np.array([-1.42064144,  0.68615248, -0.12268064, -1.56841365,  0.77278432,
                 -2.9957832 , -0.80084822, -1.5276294 , -0.96945657, -0.42762472,
                 0.06589666, 1.72620247, 2.53837067, 2.06008666, 0.67753624,
                 2.76686856, 1.38138495, 0.92817373, 1.86015938, 1.54262541])
   

   likelihood = NormalLikelihood()
   parameters = [Normal(), InvGamma()]
   model = Model(parameters, likelihood)

   sampler = MHSampler(y, model)

   sampler.sample(1000)

   params = sampler.get_chain()
