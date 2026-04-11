import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Any
from numpy_array import NumpyArray

class Matrix(NumpyArray):
    def slice(self, idx1: int, idx2: int) -> NDArray[Any]:
        return self.np_arr[:idx1, idx2:]

    def max(self, axis: int) -> NDArray[Any]:
        return np.max(self.np_arr, axis = axis)

    def arg_max(self, axis: int) -> NDArray[Any]:
        return np.argmax(self.np_arr, axis = axis)

    def random(self, seed: int, x: int, y: int) -> NDArray[Any]:
        np.random.seed(seed)
        return np.random.rand(x, y)

    def zeros(self, x: int, y: int) -> NDArray[Any]:
        return np.zeros((x, y))

    def ones(self, x: int, y: int) -> NDArray[Any]:
        return np.ones((x, y))

    def empty(self, x: int, y: int) -> NDArray[Any]:
        return np.empty((x, y))
