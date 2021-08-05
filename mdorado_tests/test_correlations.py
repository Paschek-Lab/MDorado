#!/usr/bin/env python3.8

import unittest
import numpy as np
from mdorado import correlations

class TestProgram(unittest.TestCase):
    def test_correlate(self):
        a = [1, 0, 1, 1, 1, 0, 0, 1]
        b = [1, 1, 1, 0, 1, 1, 0, 0]
        arr =np.array([0.375, 0.42857143, 0.5, 0.2, 0.25, 0.33333333, 0, 0])
        result = correlations.correlate(a, b)
        self.assertIsNone(np.testing.assert_array_almost_equal(arr, result))

    def test_autocorrelate(self):
        arr = np.array([0.53846154, 0.16666667, 0.36363636, 0.1, 0.33333333, 0, 0.28571429, 0.16666667, 0.6, 0.25, 0.66666667, 0.5, 1])
        result = correlations.correlate([1, 0, 1, 0, 1, 0, 0, 0 ,1 ,0, 1, 1, 1])
        self.assertIsNone(np.testing.assert_array_almost_equal(arr, result))

if __name__ == '__main__':
        unittest.main()

