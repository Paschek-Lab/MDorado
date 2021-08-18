import unittest
import tempfile
import os
import numpy as np
import MDAnalysis
from mdorado import msd
from mdorado.data.datafilenames import (water_topology, 
                                        water_trajectory,
                                        test_unshift,
                                        test_msd
                                            )

class TestProgram(unittest.TestCase):
    def test_unshift(self):
        u = MDAnalysis.Universe(water_topology, water_trajectory, tpr_resid_from_one=False)
        mol42grp = u.select_atoms("resname SOL and resid 42")
        mol42pos = msd.unshift(universe=u, agrp=mol42grp, dimensionskey="xyz", cms=True)
        refunshift = np.load(test_unshift)
        self.assertIsNone(np.testing.assert_array_almost_equal(refunshift, mol42pos))
    def test_msd(self):
        u = MDAnalysis.Universe(water_topology, water_trajectory, tpr_resid_from_one=False)
        hgrp = u.select_atoms("name hw")
        hpos = msd.unshift(universe=u, agrp=hgrp, dimensionskey="xyz", cms=False)
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            msd.msd(positions=hpos, dt=0.2, outfilename="msd_h.dat") 
            hmsd = np.loadtxt("msd_h.dat")
            refmsd = np.loadtxt(test_msd)
            self.assertIsNone(np.testing.assert_array_almost_equal(refmsd, hmsd))

if __name__ == '__main__':
        unittest.main()

