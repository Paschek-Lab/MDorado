import MDAnalysis
import mdorado.vectors as mvec
from mdorado.data.datafilenames import water_topology, water_trajectory

u = MDAnalysis.Universe(water_topology, water_trajectory)
ogrp = u.select_atoms("name ow")

vecmat = mvec.get_vectormatrix(universe=u, agrp=ogrp, bgrp=ogrp, pbc=True)
print(vecmat.shape)
