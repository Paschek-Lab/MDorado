import unittest
import tempfile
import os
import numpy as np
import MDAnalysis
from mdorado.lifetime import calc_lifetime
from mdorado.data.datafilenames import water_topology, water_trajectory, test_lifetime

class TestProgram(unittest.TestCase):
    def test_lifetime(self):
        u = MDAnalysis.Universe(water_topology, water_trajectory, tpr_resid_from_one=False)
        ygrp = u.select_atoms("name ow")
        xgrp = u.select_atoms("name ow and resid 15")
        hgrp = u.select_atoms("name hw and resid 15")[:1:]
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            calc_lifetime(universe=u, timestep=0.2, xgrp=xgrp, hgrp=hgrp, ygrp=ygrp, cutoff_hy=2.5, angle_cutoff=2.27, cutoff_xy=3.5)
            output = np.loadtxt("ct_0.dat")

            test_output = np.loadtxt(test_lifetime)
            self.assertIsNone(np.testing.assert_array_almost_equal(output, test_output))

if __name__ == '__main__':
    unittest.main()

