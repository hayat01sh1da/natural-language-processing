import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal

from matrix import Matrix


@pytest.fixture
def matrix():
    return Matrix([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ])


def test_array(matrix):
    assert_array_equal(
        matrix.array(),
        np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]),
    )


def test_shape(matrix):
    assert matrix.shape() == (4, 4)


def test_slice(matrix):
    assert_array_equal(matrix.slice(2, 1), np.array([[2, 3, 4], [6, 7, 8]]))


def test_sum(matrix):
    assert_array_equal(
        matrix.sum(matrix.array()),
        np.array([
            [2, 4, 6, 8],
            [10, 12, 14, 16],
            [18, 20, 22, 24],
            [26, 28, 30, 32],
        ]),
    )


def test_broadcasting_sum(matrix):
    assert_array_equal(
        matrix.sum(10),
        np.array([
            [11, 12, 13, 14],
            [15, 16, 17, 18],
            [19, 20, 21, 22],
            [23, 24, 25, 26],
        ]),
    )


def test_substract(matrix):
    mtx = np.array([
        [1, 15, 14, 8],
        [17, 9, 3, 19],
        [16, 8, 19, 8],
        [16, 3, 2, 12],
    ])
    assert_array_equal(
        matrix.substract(mtx),
        np.array([
            [0, -13, -11, -4],
            [-12, -3, 4, -11],
            [-7, 2, -8, 4],
            [-3, 11, 13, 4],
        ]),
    )


def test_multiply(matrix):
    assert_array_equal(
        matrix.multiply(matrix.array()),
        np.array([
            [1, 4, 9, 16],
            [25, 36, 49, 64],
            [81, 100, 121, 144],
            [169, 196, 225, 256],
        ]),
    )


def test_broadcasting_multiply(matrix):
    assert_array_equal(
        matrix.multiply(2),
        np.array([
            [2, 4, 6, 8],
            [10, 12, 14, 16],
            [18, 20, 22, 24],
            [26, 28, 30, 32],
        ]),
    )


def test_dot_product(matrix):
    mtx = np.array([
        [1, 15, 14, 8],
        [17, 9, 3, 19],
        [16, 8, 19, 8],
        [16, 3, 2, 12],
    ])
    assert_array_equal(
        matrix.dot_product(mtx),
        np.array([
            [147, 69, 85, 118],
            [347, 209, 237, 306],
            [547, 349, 389, 494],
            [747, 489, 541, 682],
        ]),
    )

    vec = np.array([1, 2, 2, 0])
    assert_array_equal(matrix.dot_product(vec), np.array([11, 31, 51, 71]))


def test_max(matrix):
    assert_array_equal(matrix.max(0), np.array([13, 14, 15, 16]))
    assert_array_equal(matrix.max(1), np.array([4, 8, 12, 16]))


def test_arg_max(matrix):
    assert_array_equal(matrix.arg_max(0), np.array([3, 3, 3, 3]))
    assert_array_equal(matrix.arg_max(1), np.array([3, 3, 3, 3]))


def test_numpy_sum(matrix):
    assert matrix.numpy_sum() == 136


def test_mean(matrix):
    assert matrix.mean() == 8.5


def test_random(matrix):
    assert_almost_equal(
        matrix.random(10, 4, 4),
        np.array([
            [0.7713206, 0.0207519, 0.6336482, 0.7488039],
            [0.498507, 0.2247966, 0.1980629, 0.7605307],
            [0.1691108, 0.0883398, 0.6853598, 0.9533933],
            [0.0039483, 0.5121923, 0.812621, 0.6125261],
        ]),
    )


def test_zeros(matrix):
    assert_almost_equal(matrix.zeros(4, 4), np.zeros((4, 4)))


def test_ones(matrix):
    assert_almost_equal(matrix.ones(4, 4), np.ones((4, 4)))


def test_empty(matrix):
    assert matrix.empty(4, 4).shape == (4, 4)


def test_hstack(matrix):
    mtx = np.array([
        [1, 15, 14, 8],
        [17, 9, 3, 19],
        [16, 8, 19, 8],
        [16, 3, 2, 12],
    ])
    assert_array_equal(
        matrix.hstack(mtx),
        np.array([
            [1, 2, 3, 4, 1, 15, 14, 8],
            [5, 6, 7, 8, 17, 9, 3, 19],
            [9, 10, 11, 12, 16, 8, 19, 8],
            [13, 14, 15, 16, 16, 3, 2, 12],
        ]),
    )


def test_vstack(matrix):
    mtx = np.array([
        [1, 15, 14, 8],
        [17, 9, 3, 19],
        [16, 8, 19, 8],
        [16, 3, 2, 12],
    ])
    assert_array_equal(
        matrix.vstack(mtx),
        np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [1, 15, 14, 8],
            [17, 9, 3, 19],
            [16, 8, 19, 8],
            [16, 3, 2, 12],
        ]),
    )
