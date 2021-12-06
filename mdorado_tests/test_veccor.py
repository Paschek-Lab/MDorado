import unittest
import numpy as np
import MDAnalysis
from mdorado import veccor
from mdorado.data.datafilenames import (water_topology, 
                                        water_trajectory,
                                        test_getvec,
                                        test_getnvec,
                                        test_correlvec,
                                        test_vectors,
                                        test_isocorrelvec,
                                        test_isocorrelveclg1,
                                        test_isocorrelveclg2
                                            )

class TestProgram(unittest.TestCase):
    def test_getvec(self):
        refvectors = np.loadtxt(test_getvec)
        self.assertIsNone(np.testing.assert_array_almost_equal(refvectors, vectors[42]))

    def test_getnvec(self):
        h2grp = u.select_atoms("name hw")[1::2]
        refvectors = np.loadtxt(test_getnvec)
        normalvectors = veccor.get_normal_vec(universe=u, agrp=ogrp, bgrp=hgrp, cgrp=h2grp)
        self.assertIsNone(np.testing.assert_array_almost_equal(refvectors, normalvectors[42]))

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
        vectors = veccor.get_vec(universe=u, agrp=ogrp, bgrp=hgrp)
        unittest.main()

