import MDAnalysis
from mdorado.hb_analyze import hb_analyze
from mdorado.data.datafilenames import water_topology, water_trajectory

u = MDAnalysis.Universe(water_topology, water_trajectory)
xgrp = u.select_atoms("name ow")
hgrp = u.select_atoms("name hw")[::2]

hb_analyze(universe=u, xgrp=xgrp, hgrp=hgrp, rmin=1.5, rmax=5, cosalphamin=-1, cosalphamax=1, bins=50)
