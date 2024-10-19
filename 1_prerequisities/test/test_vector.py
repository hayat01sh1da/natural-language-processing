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
        assert_array_equal(np.array([1, 2, 3, 4, 5]), self.vector.array())

    def test_shape(self):
        self.assertEqual((5,), self.vector.shape())

    def test_slice(self):
        assert_array_equal(np.array([ 1, 2, 3]), self.vector.slice(3))

    def test_sum(self):
        assert_array_equal(
            np.array([2, 4, 6, 8, 10]),
            self.vector.sum(self.vector.array())
        )

    def test_broadcasting_sum(self):
        assert_array_equal(
            np.array([11, 12, 13, 14, 15]),
            self.vector.sum(10)
        )

    def test_substract(self):
        vec = np.array([1, 2, 2, 0, 1])
        assert_array_equal(
            np.array([0, 0, 1, 4, 4]),
            self.vector.substract(vec)
        )

    def test_multiply(self):
        assert_array_equal(
            np.array([1, 4, 9, 16, 25]),
            self.vector.multiply(self.vector.array())
        )

    def test_broadcasting_multiply(self):
        assert_array_equal(
            np.array([2, 4, 6, 8, 10]),
            self.vector.multiply(2)
        )

    def test_dot_product(self):
        vec = np.array([1, 2, 2, 0, 1])
        self.assertEqual(16, self.vector.dot_product(vec))

    def test_max(self):
        self.assertEqual(5, self.vector.max())

    def test_arg_max(self):
        self.assertEqual(4, self.vector.arg_max())

    def test_numpy_sum(self):
        self.assertEqual(15, self.vector.numpy_sum())

    def test_mean(self):
        self.assertEqual(3, self.vector.mean())

    def test_exp(self):
        assert_almost_equal(np.array([2.7182818, 7.3890561, 20.0855369, 54.59815, 148.4131591]), self.vector.exp())

    def test_random(self):
        assert_almost_equal(np.array([0.7713206, 0.0207519, 0.6336482, 0.7488039, 0.498507]), self.vector.random(10, 5))

    def test_zeros(self):
        assert_almost_equal(np.array([0, 0, 0, 0, 0]), self.vector.zeros(5))

    def test_ones(self):
        assert_almost_equal(np.array([1, 1, 1, 1, 1]), self.vector.ones(5))

    def test_empty(self):
        self.assertEqual((5,), self.vector.empty(5).shape)

    def test_hstack(self):
        vec = np.array([1, 2, 2, 0, 1])
        assert_array_equal(
            np.array([1, 2, 3, 4, 5, 1, 2, 2, 0, 1]),
            self.vector.hstack(vec)
        )

    def test_vstack(self):
        vec = np.array([1, 2, 2, 0, 1])
        assert_array_equal(
            np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 2, 0, 1],
        ]),
            self.vector.vstack(vec)
        )

if __name__ == '__main__':
    unittest.main()
