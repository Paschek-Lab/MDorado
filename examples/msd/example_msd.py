import MDAnalysis
from mdorado import msd
from mdorado.data.datafilenames import water_topology, water_trajectory

u = MDAnalysis.Universe(water_topology, water_trajectory)
solgrp = u.select_atoms("resname SOL")
hgrp = u.select_atoms("name hw")
dt = 0.2

cmspos = msd.unwrap(universe=u, agrp=solgrp, dimensionskey="xyz", cms=True)
msd.msd(positions=cmspos, dt=dt, outfilename="msd_cms.dat")

hpos = msd.unwrap(universe=u, agrp=hgrp, dimensionskey="xyz", cms=False)
msd.msd(positions=hpos, dt=dt, outfilename="msd_h.dat")
