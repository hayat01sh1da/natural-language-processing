import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Any
from numpy_array import NumpyArray

class Vector(NumpyArray):
    def slice(self, idx: int) -> NDArray[Any]:
        return self.np_arr[:idx]

    def max(self) -> Any:
        return np.max(self.np_arr)

    def arg_max(self) -> Any:
        return np.argmax(self.np_arr)

    def random(self, seed: int, n: int) -> NDArray[Any]:
        np.random.seed(seed)
        return np.random.rand(n)

    def zeros(self, n: int) -> NDArray[Any]:
        return np.zeros((n,))

    def ones(self, n: int) -> NDArray[Any]:
        return np.ones((n,))

    def empty(self, n: int) -> NDArray[Any]:
        return np.empty((n,))
