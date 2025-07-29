import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sys
import os
import shutil
import glob
sys.path.append('./src')
from vector import Vector

class TestVector(unittest.TestCase):
    def setUp(self):
        vec           = [1, 2, 3, 4, 5]
        self.vector   = Vector(vec)
        self.pycaches = glob.glob(os.path.join('.', '**', '__pycache__'), recursive = True)

    def tearDown(self):
        for pycache in self.pycaches:
            if os.path.exists(pycache):
                shutil.rmtree(pycache)

    def test_array(self):
        assert_array_equal(self.vector.array(), np.array([1, 2, 3, 4, 5]))

    def test_shape(self):
        self.assertEqual(self.vector.shape(), (5,))

    def test_slice(self):
        assert_array_equal(self.vector.slice(3), np.array([ 1, 2, 3]))

    def test_sum(self):
        assert_array_equal(
            self.vector.sum(self.vector.array()),
            np.array([2, 4, 6, 8, 10])
        )

    def test_broadcasting_sum(self):
        assert_array_equal(
            self.vector.sum(10),
            np.array([11, 12, 13, 14, 15])
        )

    def test_substract(self):
        vec = np.array([1, 2, 2, 0, 1])
        assert_array_equal(
            self.vector.substract(vec),
            np.array([0, 0, 1, 4, 4])
        )

    def test_multiply(self):
        assert_array_equal(
            self.vector.multiply(self.vector.array()),
            np.array([1, 4, 9, 16, 25])
        )

    def test_broadcasting_multiply(self):
        assert_array_equal(
            self.vector.multiply(2),
            np.array([2, 4, 6, 8, 10])
        )

    def test_dot_product(self):
        vec = np.array([1, 2, 2, 0, 1])
        self.assertEqual(self.vector.dot_product(vec), 16)

    def test_max(self):
        self.assertEqual(self.vector.max(), 5)

    def test_arg_max(self):
        self.assertEqual(self.vector.arg_max(), 4)

    def test_numpy_sum(self):
        self.assertEqual(self.vector.numpy_sum(), 15)

    def test_mean(self):
        self.assertEqual(self.vector.mean(), 3)

    def test_exp(self):
        assert_almost_equal(self.vector.exp(), np.array([2.71828183, 7.3890561, 20.08553692, 54.59815003, 148.4131591]))

    def test_random(self):
        assert_almost_equal(self.vector.random(10, 5), np.array([0.7713206, 0.0207519, 0.6336482, 0.7488039, 0.498507]))

    def test_zeros(self):
        assert_almost_equal(self.vector.zeros(5), np.array([0, 0, 0, 0, 0]))

    def test_ones(self):
        assert_almost_equal(self.vector.ones(5), np.array([1, 1, 1, 1, 1]))

    def test_empty(self):
        self.assertEqual(self.vector.empty(5).shape, (5,))

    def test_hstack(self):
        vec = np.array([1, 2, 2, 0, 1])
        assert_array_equal(
            self.vector.hstack(vec),
            np.array([1, 2, 3, 4, 5, 1, 2, 2, 0, 1])
        )

    def test_vstack(self):
        vec = np.array([1, 2, 2, 0, 1])
        assert_array_equal(
            self.vector.vstack(vec),
            np.array([
                [1, 2, 3, 4, 5],
                [1, 2, 2, 0, 1],
            ])
        )

if __name__ == '__main__':
    unittest.main()
