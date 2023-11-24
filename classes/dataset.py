import numpy as np


class Dataset:
    _y = []
    def __init__(self, y:list|np.ndarray) -> None:
        self._y = y

    def __getitem__(self, id) -> int|float:
        try:
            assert id < len(self._y)
        except:
            raise IndexError(f"Tried getting dataset {id} out of range for length {len(self._y)}")
        return self._y[id]
    

class DPMDataset:
    _y = []
    _s = []
    def __init__(self, y:list|np.ndarray=None, s:list|np.ndarray=None) -> None:
        self._y = y
        self._s = s

    def set_y(self, y:list|np.ndarray):
        if isinstance(y, np.ndarray):
            y = list(y)
        self._y = y.copy()
        
    def set_s(self, s:list|np.ndarray):
        if isinstance(s, np.ndarray):
            s = list(s)
        self._s = s.copy()

    def get_y(self) -> list:
        return self._y

    def get_s(self) -> list:
        return self._s
    
    def get_cluster_data(self, id) -> list:
        ## Given a cluster id, check if it's in the list and then return 
        # the data pertaining to that cluster
        clusters = set(self._s)
        try:
            assert id in clusters
        except:
            raise LookupError(f"{id} is not in {clusters}")
        
        return_data = []

        for i in range(len(self._s)):
            if self._s[i] == id:
                return_data.append(self._y[i])

        return return_data

    def calculate_weights(self) -> list:
        c, n = self.count_clusters()
        totals = sum(n)

        for i in range(len(n)):
            n_new = n[i] / totals
            n[i] = n_new

        return n
        
    def count_clusters(self) -> list[list]:
        c_j, n_j = np.unique(self._s, return_counts=True)
        c_j = list(c_j)
        n_j = list(n_j)    
        return [c_j, n_j]


    def __getitem__(self, id) -> list:
        try:
            assert id < len(self._y) and len(self._y) == len(self._s)
        except:
            raise IndexError(f"Tried getting dataset {id} out of range for length {len(self._y)}")
        return [self._y[id], self._s[id]]
    
    def __len__(self) -> int:
        return len(self._y)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        s = ""
        for i in range(len(self._y)):
            s += str(self.__getitem__(i))
            s += "\n"
        return s