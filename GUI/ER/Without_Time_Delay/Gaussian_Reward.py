import numpy as np

class Gaussian_Reward():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, size= 1):
        return np.random.normal(loc=self.mean,
                                scale=self.std,
                                size=size )


