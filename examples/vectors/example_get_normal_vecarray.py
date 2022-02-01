import MDAnalysis
import mdorado.vectors as mvec
from mdorado.data.datafilenames import water_topology, water_trajectory

u = MDAnalysis.Universe(water_topology, water_trajectory)
ogrp = u.select_atoms("name ow")
h1grp = u.select_atoms("name hw")[::2]
h2grp = u.select_atoms("name hw")[1::2]

normalvectorarray = mvec.get_normal_vecarray(universe=u, agrp=ogrp, bgrp=h2grp, cgrp=h2grp, pbc=False)
print(normalvectorarray.shape)
