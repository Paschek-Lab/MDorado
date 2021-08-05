import MDAnalysis
from mdorado.lifetime import calc_lifetime
from mdorado.data.datafilenames import water_topology, water_trajectory

universe = MDAnalysis.Universe(water_topology, water_trajectory)

xgrp = universe.select_atoms("name ow")[:20]
hgrp = universe.select_atoms("name hw")[:40:2]
ygrp = universe.select_atoms("name ow")

calc_lifetime(universe=universe, timestep=0.2, xgrp=xgrp, hgrp=hgrp, ygrp=ygrp, cutoff_hy=2.5, angle_cutoff=2.27, cutoff_xy=3.5)
