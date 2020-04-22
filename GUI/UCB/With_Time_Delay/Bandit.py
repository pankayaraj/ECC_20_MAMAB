import numpy as np


class Bandit():

    def __init__(self, index, mean, std):
        self.index = index
        self.mean = mean
        self.std = std

    def sample(self, size= 1):
        return np.random.normal(loc=self.mean,
                                scale=self.std,
                                size=size )

