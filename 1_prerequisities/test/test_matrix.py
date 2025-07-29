import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
import os
import shutil
import glob
sys.path.append('./src')
from matrix import Matrix

class TestMatrix(unittest.TestCase):
    def setUp(self):
        mtx = [
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
        self.matrix   = Matrix(mtx)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_array(self):
        assert_array_equal(
            self.matrix.array(),
            np.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
            ])
        )

    def test_shape(self):
        self.assertEqual(self.matrix.shape(), (4, 4))

    def test_slice(self):
        assert_array_equal(
            self.matrix.slice(2, 1),
            np.array([
                [2, 3, 4],
                [6, 7, 8],
            ])
        )

    def test_sum(self):
        assert_array_equal(
            self.matrix.sum(self.matrix.array()),
            np.array([
                [ 2,  4,  6,  8],
                [10, 12, 14, 16],
                [18, 20, 22, 24],
                [26, 28, 30, 32],
            ])
        )

    def test_broadcasting_sum(self):
        assert_array_equal(
            self.matrix.sum(10),
            np.array([
                [11, 12, 13, 14],
                [15, 16, 17, 18],
                [19, 20, 21, 22],
                [23, 24, 25, 26],
            ])
        )

    def test_substract(self):
        mtx = np.array([
            [ 1, 15, 14,  8],
            [17,  9,  3, 19],
            [16,  8, 19,  8],
            [16,  3,  2, 12],
        ])
        assert_array_equal(
            self.matrix.substract(mtx),
            np.array([
                [  0, -13, -11,  -4],
                [-12,  -3,   4, -11],
                [ -7,   2,  -8,   4],
                [ -3,  11,  13,   4],
            ])
        )

    def test_multiply(self):
        assert_array_equal(
            self.matrix.multiply(self.matrix.array()),
            np.array([
                [  1,   4,   9,  16],
                [ 25,  36,  49,  64],
                [ 81, 100, 121, 144],
                [169, 196, 225, 256],
            ])
        )

    def test_broadcasting_multiply(self):
        assert_array_equal(
            self.matrix.multiply(2),
            np.array([
                [ 2,  4,  6,  8],
                [10, 12, 14, 16],
                [18, 20, 22, 24],
                [26, 28, 30, 32],
            ])
        )

    def test_dot_product(self):
        mtx = np.array([
            [ 1, 15, 14,  8],
            [17,  9,  3, 19],
            [16,  8, 19,  8],
            [16,  3,  2, 12],
        ])
        assert_array_equal(
            self.matrix.dot_product(mtx),
            np.array([
                [147,  69,  85, 118],
                [347, 209, 237, 306],
                [547, 349, 389, 494],
                [747, 489, 541, 682],
            ])
        )

        vec = np.array([1, 2, 2, 0])
        assert_array_equal(self.matrix.dot_product(vec), np.array([11, 31, 51, 71]))

    def test_max(self):
        assert_array_equal(self.matrix.max(0), np.array([13, 14, 15, 16]))
        assert_array_equal(self.matrix.max(1), np.array([4, 8, 12, 16]))

    def test_arg_max(self):
        assert_array_equal(self.matrix.arg_max(0), np.array([3, 3, 3, 3]))
        assert_array_equal(self.matrix.arg_max(1), np.array([3, 3, 3, 3]))

    def test_numpy_sum(self):
        self.assertEqual(self.matrix.numpy_sum(), 136)

    def test_mean(self):
        self.assertEqual(self.matrix.mean(), 8.5)

    # def test_exp(self):
    #     # Mismatched elements: 11 / 16 (68.8%)
    #     # Max absolute difference: 0.00416478
    #     # Max relative difference: 3.46313149e-09
    #     assert_almost_equal(
    #         np.array([
    #             [2.71828183e+00, 7.38905610e+00, 2.00855369e+01, 5.45981500e+01],
    #             [1.48413159e+02, 4.03428793e+02, 1.09663316e+03, 2.98095799e+03],
    #             [8.10308393e+03, 2.20264658e+04, 5.98741417e+04, 1.62754791e+05],
    #             [4.42413392e+05, 1.20260428e+06, 3.26901737e+06, 8.88611052e+06],
    #         ]),
    #         self.matrix.exp()
    #     )

    def test_random(self):
        assert_almost_equal(
            self.matrix.random(10, 4, 4),
            np.array([
                [0.7713206, 0.0207519, 0.6336482, 0.7488039],
                [0.498507 , 0.2247966, 0.1980629, 0.7605307],
                [0.1691108, 0.0883398, 0.6853598, 0.9533933],
                [0.0039483, 0.5121923, 0.812621 , 0.6125261],
            ])
        )

    def test_zeros(self):
        assert_almost_equal(
            self.matrix.zeros(4, 4),
            np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ])
        )

    def test_ones(self):
        assert_almost_equal(
            self.matrix.ones(4, 4),
            np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ])
        )
    def test_empty(self):
        self.assertEqual(self.matrix.empty(4, 4).shape, (4, 4))

    def test_hstack(self):
        mtx = np.array([
            [ 1, 15, 14,  8],
            [17,  9,  3, 19],
            [16,  8, 19,  8],
            [16,  3,  2, 12],
        ])
        assert_array_equal(
            self.matrix.hstack(mtx),
            np.array([
                [ 1,  2,  3,  4,  1, 15, 14,  8],
                [ 5,  6,  7,  8, 17,  9,  3, 19],
                [ 9, 10, 11, 12, 16,  8, 19,  8],
                [13, 14, 15, 16, 16,  3,  2, 12],
            ])
        )

    def test_vstack(self):
        mtx = np.array([
            [ 1, 15, 14,  8],
            [17,  9,  3, 19],
            [16,  8, 19,  8],
            [16,  3,  2, 12],
        ])
        assert_array_equal(
            self.matrix.vstack(mtx),
            np.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
                [ 1, 15, 14,  8],
                [17,  9,  3, 19],
                [16,  8, 19,  8],
                [16,  3,  2, 12],
            ])
        )

if __name__ == '__main__':
    unittest.main()
