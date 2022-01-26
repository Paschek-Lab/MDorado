import unittest
import numpy as np
import MDAnalysis
from mdorado import vectors as mvec
from mdorado import veccor
from mdorado.data.datafilenames import (water_topology, 
                                        water_trajectory,
                                        test_correlvec,
                                        test_vectors,
                                        test_isocorrelvec,
                                        test_isocorrelveclg1,
                                        test_isocorrelveclg2
                                            )

class TestProgram(unittest.TestCase):
    def test_correlvec(self):
        ts, correl = veccor.correlvec(vectors, refvec=[1,1,1], nlegendre=2, dt=dt)
        refcorrel = np.loadtxt(test_correlvec)
        self.assertIsNone(np.testing.assert_array_almost_equal(refcorrel, np.array([ts, correl]).T))

    def test_isocorrelvec(self):
        isovectors = np.load(test_vectors)
        ts, correl = veccor.isocorrelvec(isovectors, nlegendre=2, dt=dt)
        refcorrel = np.loadtxt(test_isocorrelvec)
        self.assertIsNone(np.testing.assert_array_almost_equal(refcorrel, np.array([ts, correl]).T))

    def test_isocorrelveclg1(self):
        ts, correl = veccor.isocorrelveclg1(vectors, dt=dt)
        refcorrel = np.loadtxt(test_isocorrelveclg1)
        self.assertIsNone(np.testing.assert_array_almost_equal(refcorrel, np.array([ts, correl]).T))

    def test_isocorrelveclg2(self):
        ts, correl = veccor.isocorrelveclg2(vectors, dt=dt)
        refcorrel = np.loadtxt(test_isocorrelveclg2)
        self.assertIsNone(np.testing.assert_array_almost_equal(refcorrel, np.array([ts, correl]).T))


if __name__ == '__main__':
        u = MDAnalysis.Universe(water_topology, water_trajectory, tpr_resid_from_one=False)
        ogrp = u.select_atoms("name ow")
        hgrp = u.select_atoms("name hw")[::2]
        dt=0.2
        vectors = mvec.norm_vecarray(mvec.get_vecarray(universe=u, agrp=ogrp, bgrp=hgrp))[0]
        unittest.main()

