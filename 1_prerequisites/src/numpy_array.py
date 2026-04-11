import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Any

class NumpyArray:
    def __init__(self, arr: ArrayLike) -> None:
        self.np_arr: NDArray[Any] = np.array(arr)

    def array(self) -> NDArray[Any]:
        return self.np_arr

    def shape(self) -> tuple[int, ...]:
        return self.np_arr.shape

    def sum(self, arr: ArrayLike) -> NDArray[Any]:
        return self.np_arr + arr

    def substract(self, arr: ArrayLike) -> NDArray[Any]:
        return self.np_arr - arr

    def multiply(self, arr: ArrayLike) -> NDArray[Any]:
        return self.np_arr * arr

    def dot_product(self, arr: ArrayLike) -> NDArray[Any]:
        return np.dot(self.np_arr, arr)

    def numpy_sum(self) -> Any:
        return np.sum(self.np_arr)

    def mean(self) -> Any:
        return np.mean(self.np_arr)

    def exp(self) -> NDArray[Any]:
        return np.exp(self.np_arr)

    def hstack(self, arr: ArrayLike) -> NDArray[Any]:
        return np.hstack((self.np_arr, arr))

    def vstack(self, arr: ArrayLike) -> NDArray[Any]:
        return np.vstack((self.np_arr, arr))
