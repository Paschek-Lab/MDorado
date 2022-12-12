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
            
            sitesite_intra = Gofr(universe=u, agrp=ogrp, bgrp=hgrp, rmin=0.0, rmax=6, bins=100,
                            mode="site-site", outfilename="gofr_ss_intra.dat", count_only='intra')
            
            sitesite_inter = Gofr(universe=u, agrp=ogrp, bgrp=hgrp, rmin=0.0, rmax=6, bins=100,
                            mode="site-site", outfilename="gofr_ss_inter.dat",count_only='inter')
            
            sitesite = Gofr(universe=u, agrp=hgrp, bgrp=ogrp, rmin=1.0, rmax=6, bins=100,
                            mode="site-site", outfilename="gofr_ss.dat")
            cmscms = Gofr(universe=u, agrp=watergrp, bgrp=watergrp, rmin=1.0, rmax=6, bins=100,
                          mode="cms-cms", outfilename="gofr_cc.dat")
            sitecms = Gofr(universe=u, agrp=hgrp, bgrp=watergrp, rmin=1.0, rmax=6, bins=100,
                           mode="site-cms", outfilename="gofr_sc.dat")

            #TODO: Calculate Reference Data for Intra and Inter site-site gofr
            
            #gss_intra = np.loadtxt(test_gofr_ss_intra, unpack=True)
            #gss_inter = np.loadtxt(test_gofr_ss_inter, unpack=True)
            gss = np.loadtxt(test_gofr_ss, unpack=True)
            gcc = np.loadtxt(test_gofr_cc, unpack=True)
            gsc = np.loadtxt(test_gofr_sc, unpack=True)

            resss_intra = np.array([sitesite_intra.rdat, sitesite_intra.hist, sitesite_intra.annn, sitesite_intra.bnnn])
            resss_inter = np.array([sitesite_inter.rdat, sitesite_inter.hist, sitesite_inter.annn, sitesite_inter.bnnn])
            resss = np.array([sitesite.rdat, sitesite.hist, sitesite.annn, sitesite.bnnn])
            rescc = np.array([cmscms.rdat, cmscms.hist, cmscms.annn, cmscms.bnnn])
            ressc = np.array([sitecms.rdat, sitecms.hist, sitecms.annn, sitecms.bnnn])
            
            #self.assertIsNone(np.testing.assert_array_almost_equal(resss_intra, gss_intra))
            #self.assertIsNone(np.testing.assert_array_almost_equal(resss_inter, gss_inter))
                        
            self.assertEqual(sitesite_intra.annn[-1], 1, 'Number of oxygen atoms should converge to 1 for the intramolecular case')
            self.assertEqual(sitesite_intra.bnnn[-1], 2, 'Number of hydrogen atoms should converge to 2 for the intramolecular case')
            self.assertIsNone(np.testing.assert_array_almost_equal(resss, gss), 'site-site test set should be almost equal to reference set')
            self.assertIsNone(np.testing.assert_array_almost_equal(rescc, gcc), 'cms-cms test set should be almost equal to reference set')
            self.assertIsNone(np.testing.assert_array_almost_equal(ressc, gsc), 'site-cms test set should be almost equal to reference set')
    
    

if __name__ == '__main__':
    unittest.main()
