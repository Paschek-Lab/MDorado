import unittest
import numpy as np
import MDAnalysis
from mdorado import vectors as mvec
from mdorado.data.datafilenames import (water_topology, 
                                        water_trajectory,
                                        test_getvecarray,
                                        test_getnormvecarray,
                                        test_getvecmatrix
                                            )

class TestProgram(unittest.TestCase):
    def test_pbc_vecarray(self):
        refvectors = np.array([
            [1, -1,  3],
            [4,  1,  1],
            [4, -2,  2],
            [3,  0, -1]
            ])
        pbcvecarray = mvec.pbc_vecarray(vecarray, box)
        self.assertIsNone(np.testing.assert_array_almost_equal(refvectors, pbcvecarray))

    def test_norm_vecarray(self):
        refnormarray = np.array([
            [0.88929729, 0.32338083, -0.32338083],
            [0.44444444, 0.11111111, 0.88888889],
            [-0.63599873, -0.74199852, 0.21199958],
            [0.9486833, 0., -0.31622777]
            ])
        refnorm = np.array([12.36931688,9., 9.43398113, 3.16227766])
        normarray, norm = mvec.norm_vecarray(vecarray)
        self.assertIsNone(np.testing.assert_array_almost_equal(refnormarray, normarray))
        self.assertIsNone(np.testing.assert_array_almost_equal(refnorm, norm))

    def test_vectormatrix(self):
        refmat = np.array([
            [[ 0, 0, 0],
            [-7, -3., 12],
            [-17, -11, 6],
            [-8, -4, 3]],

            [[7, 3, -12],
            [0, 0, 0],
            [-10, -8, -6],
            [-1, -1, -9]],

            [[17, 11, -6],
            [10, 8, 6],
            [0, 0, 0],
            [9, 7, -3]],

            [[8, 4, -3],
            [1, 1, 9],
            [-9, -7, 3],
            [0, 0, 0]]
            ])
        vecmat = mvec.vectormatrix(vecarray, vecarray)
        self.assertIsNone(np.testing.assert_array_almost_equal(refmat, vecmat))

    def test_get_vecarray(self):
        vectors = mvec.get_vecarray(universe=u, agrp=ogrp, bgrp=hgrp)
        refvectors = np.load(test_getvecarray)
        self.assertIsNone(np.testing.assert_array_almost_equal(refvectors, vectors[42]))

    def test_get_normalvecarray(self):
        refvectors = np.load(test_getnormvecarray)
        normalvectors = mvec.get_normal_vecarray(universe=u, agrp=ogrp, bgrp=hgrp, cgrp=h2grp)
        self.assertIsNone(np.testing.assert_array_almost_equal(refvectors, normalvectors[42]))
    
    def test_get_vectormatrix(self):
        refmatrix = np.load(test_getvecmatrix)
        vecmatrix = mvec.get_vectormatrix(universe=u, agrp=ogrp, bgrp=ogrp)
        self.assertIsNone(np.testing.assert_array_almost_equal(refmatrix, vecmatrix[42][47]))

if __name__ == '__main__':
        u = MDAnalysis.Universe(water_topology, water_trajectory, tpr_resid_from_one=False)
        ogrp = u.select_atoms("name ow")
        hgrp = u.select_atoms("name hw")[::2]
        h2grp = u.select_atoms("name hw")[1::2]
        dt=0.2
        box = [10,5,7]
        vecarray = np.array([
            [11,4,-4],
            [4,1,8],
            [-6,-7, 2],
            [3,0, -1]
            ])
        unittest.main()

