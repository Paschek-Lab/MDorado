import unittest
import tempfile
import os
import numpy as np
import MDAnalysis
from mdorado.gofr import Gofr
from mdorado.data.datafilenames import (water_topology, 
                                        water_trajectory,
                                        test_gofr_ss,
                                        test_gofr_cc,
                                        test_gofr_sc
                                            )

class TestProgram(unittest.TestCase):
    def test_gofr(self):
        u = MDAnalysis.Universe(water_topology, water_trajectory)
        hgrp = u.select_atoms("name hw")
        ogrp = u.select_atoms("name ow")
        watergrp = u.select_atoms("resname sol")
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            sitesite = Gofr(universe=u, agrp=hgrp, bgrp=ogrp, rmin=1.0, rmax=6, bins=100, mode="site-site", outfilename="gofr_ss.dat")
            cmscms = Gofr(universe=u, agrp=watergrp, bgrp=watergrp, rmin=1.0, rmax=6, bins=100, mode="cms-cms", outfilename="gofr_cc.dat")
            sitecms = Gofr(universe=u, agrp=hgrp, bgrp=watergrp, rmin=1.0, rmax=6, bins=100, mode="site-cms", outfilename="gofr_sc.dat")
        
            gss = np.loadtxt(test_gofr_ss, unpack=True)
            gcc = np.loadtxt(test_gofr_cc, unpack=True)
            gsc = np.loadtxt(test_gofr_sc, unpack=True)

            resss = np.array([sitesite.rdat, sitesite.hist, sitesite.annn, sitesite.bnnn])
            rescc = np.array([cmscms.rdat, cmscms.hist, cmscms.annn, cmscms.bnnn])
            ressc = np.array([sitecms.rdat, sitecms.hist, sitecms.annn, sitecms.bnnn])
            self.assertIsNone(np.testing.assert_array_almost_equal(resss, gss))
            self.assertIsNone(np.testing.assert_array_almost_equal(rescc, gcc))
            self.assertIsNone(np.testing.assert_array_almost_equal(ressc, gsc))

if __name__ == '__main__':
        unittest.main()

