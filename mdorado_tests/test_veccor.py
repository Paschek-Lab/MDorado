import unittest
import tempfile
import os
import numpy as np
import MDAnalysis
from mdorado import veccor
from mdorado.data.datafilenames import (water_topology, 
                                        water_trajectory,
                                        test_getvec,
                                        test_getnvec,
                                        test_correlvec,
                                        test_isocorrelvec,
                                        test_isocorrelveclg1,
                                        test_isocorrelveclg2
                                            )

class TestProgram(unittest.TestCase):
    def test_getvec(self):
        u = MDAnalysis.Universe(water_topology, water_trajectory, tpr_resid_from_one=False)
        ogrp = u.select_atoms("name ow")[42]
        hgrp = u.select_atoms("name hw")[84]
        refvectors = np.load(test_getvec)
        vectors = veccor.get_vec(universe=u, agrp=ogrp, bgrp=hgrp)
        self.assertIsNone(np.testing.assert_array_almost_equal(refvectors, mol42pos))

if __name__ == '__main__':
        unittest.main()

