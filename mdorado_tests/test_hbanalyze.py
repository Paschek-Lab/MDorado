import unittest
import tempfile
import os
import numpy as np
import MDAnalysis
from mdorado.hb_analyze import hb_analyze
from mdorado.data.datafilenames import water_topology, water_trajectory, test_hbanalyze

class TestProgram(unittest.TestCase):
    def test_hbanalyze(self):
        u = MDAnalysis.Universe(water_topology, water_trajectory)
        xgrp = u.select_atoms("name ow")
        hgrp = u.select_atoms("name hw")[::2]
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            hb_analyze(universe=u, xgrp=xgrp, hgrp=hgrp, rmin=1.5, rmax=5, cosalphamin=-1, cosalphamax=1, bins=50, outfilename="test.dat")
            matrix = np.loadtxt("test.dat")

            test_matrix = np.loadtxt(test_hbanalyze)
            self.assertIsNone(np.testing.assert_array_almost_equal(matrix, test_matrix))

if __name__ == '__main__':
        unittest.main()

