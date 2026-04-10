import numpy as np
from numpy_array import NumpyArray

class Matrix(NumpyArray):
    def slice(self, idx1, idx2):
        return self.np_arr[:idx1, idx2:]

    def max(self, axis):
        return np.max(self.np_arr, axis = axis)

    def arg_max(self, axis):
        return np.argmax(self.np_arr, axis = axis)

    def random(self, seed, x, y):
        np.random.seed(seed)
        return np.random.rand(x, y)

    def zeros(self, x, y):
        return np.zeros((x, y))

    def ones(self, x, y):
        return np.ones((x, y))

    def empty(self, x, y):
        return np.empty((x, y))
