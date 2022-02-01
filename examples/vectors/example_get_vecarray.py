import MDAnalysis
import mdorado.vectors as mvec
from mdorado.data.datafilenames import water_topology, water_trajectory

u = MDAnalysis.Universe(water_topology, water_trajectory)
ogrp = u.select_atoms("name ow")
hgrp = u.select_atoms("name hw")[::2]

vectorarray = mvec.get_vecarray(universe=u, agrp=ogrp, bgrp=hgrp, pbc=False)
print(vectorarray.shape)
