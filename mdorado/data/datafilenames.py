from pkg_resources import resource_filename

water_topology = resource_filename(__name__, 'water.tpr')
water_trajectory = resource_filename(__name__, 'water.xtc')
test_gofr_ss = resource_filename(__name__, 'gofr_ss.dat')
test_gofr_sc = resource_filename(__name__, 'gofr_sc.dat')
test_gofr_cc = resource_filename(__name__, 'gofr_cc.dat')
test_hbanalyze = resource_filename(__name__, 'hb_analyze.dat')
test_lifetime = resource_filename(__name__, 'lifetime_test.dat')
test_unwrap = resource_filename(__name__, 'msd_molpos.npy')
test_msd = resource_filename(__name__, 'msd_h.dat')
test_correlvec = resource_filename(__name__, 'anisolg2.dat')
test_getvecarray = resource_filename(__name__, 'vecarray42.npy')
test_getnormvecarray = resource_filename(__name__, 'normal_vecarray42.npy')
test_getvecmatrix = resource_filename(__name__, 'vecmatrix4247.npy')
test_vectors = resource_filename(__name__, 'vectors_isocorrel.npy')
test_isocorrelvec = resource_filename(__name__, 'slow_isolg2.dat')
test_isocorrelveclg1 = resource_filename(__name__, 'fast_isolg1.dat')
test_isocorrelveclg2 = resource_filename(__name__, 'fast_isolg2.dat')

del resource_filename
