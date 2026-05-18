import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal

from vector import Vector


@pytest.fixture
def vector():
    return Vector([1, 2, 3, 4, 5])


def test_array(vector):
    assert_array_equal(vector.array(), np.array([1, 2, 3, 4, 5]))


def test_shape(vector):
    assert vector.shape() == (5,)


def test_slice(vector):
    assert_array_equal(vector.slice(3), np.array([1, 2, 3]))


def test_sum(vector):
    assert_array_equal(vector.sum(vector.array()), np.array([2, 4, 6, 8, 10]))


def test_broadcasting_sum(vector):
    assert_array_equal(vector.sum(10), np.array([11, 12, 13, 14, 15]))


def test_substract(vector):
    vec = np.array([1, 2, 2, 0, 1])
    assert_array_equal(vector.substract(vec), np.array([0, 0, 1, 4, 4]))


def test_multiply(vector):
    assert_array_equal(vector.multiply(vector.array()),
                       np.array([1, 4, 9, 16, 25]))


def test_broadcasting_multiply(vector):
    assert_array_equal(vector.multiply(2), np.array([2, 4, 6, 8, 10]))


def test_dot_product(vector):
    vec = np.array([1, 2, 2, 0, 1])
    assert vector.dot_product(vec) == 16


def test_max(vector):
    assert vector.max() == 5


def test_arg_max(vector):
    assert vector.arg_max() == 4


def test_numpy_sum(vector):
    assert vector.numpy_sum() == 15


def test_mean(vector):
    assert vector.mean() == 3


def test_exp(vector):
    assert_almost_equal(vector.exp(), np.array(
        [2.71828183, 7.3890561, 20.08553692, 54.59815003, 148.4131591]), )


def test_random(vector):
    assert_almost_equal(
        vector.random(10, 5),
        np.array([0.7713206, 0.0207519, 0.6336482, 0.7488039, 0.498507]),
    )


def test_zeros(vector):
    assert_almost_equal(vector.zeros(5), np.zeros(5))


def test_ones(vector):
    assert_almost_equal(vector.ones(5), np.ones(5))


def test_empty(vector):
    assert vector.empty(5).shape == (5,)


def test_hstack(vector):
    vec = np.array([1, 2, 2, 0, 1])
    assert_array_equal(vector.hstack(vec), np.array(
        [1, 2, 3, 4, 5, 1, 2, 2, 0, 1]))


def test_vstack(vector):
    vec = np.array([1, 2, 2, 0, 1])
    assert_array_equal(
        vector.vstack(vec),
        np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 2, 0, 1],
        ]),
    )
