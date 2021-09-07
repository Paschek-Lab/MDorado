import MDAnalysis
from mdorado.gofr import Gofr
from mdorado.data.datafilenames import water_topology, water_trajectory

u = MDAnalysis.Universe(water_topology, water_trajectory)
hgrp = u.select_atoms("name hw")
ogrp = u.select_atoms("name ow")
watergrp = u.select_atoms("resname SOL")

sitesite = Gofr(universe=u, agrp=hgrp, bgrp=ogrp, rmin=1.0, rmax=6, bins=100, mode="site-site", outfilename="h_o.dat")
cmscms = Gofr(universe=u, agrp=watergrp, bgrp=watergrp, rmin=1.0, rmax=6, bins=100, mode="cms-cms", outfilename="cms_cms.dat")
sitecms = Gofr(universe=u, agrp=hgrp, bgrp=watergrp, rmin=1.0, rmax=6, bins=100, mode="site-cms", outfilename="h_cms.dat")
