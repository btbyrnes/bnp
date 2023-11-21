import numpy as np
from classes.dataset import DPMDataset


if __name__ == "__main__":
    y = np.array([-1.42064144,  0.68615248, -0.12268064, -1.56841365,  0.77278432,
                 -2.9957832 , -0.80084822, -1.5276294 , -0.96945657, -0.42762472,
                 0.06589666, 1.72620247, 2.53837067, 2.06008666])
    s = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 2, 2])

    data = DPMDataset(y, s)

    

    print(data.get_cluster_data(1))
    print(data.count_clusters())

    print(data.calculate_weights())