import MDAnalysis
from mdorado import vectors as mvec
from mdorado import veccor
from mdorado.data.datafilenames import water_topology, water_trajectory

import numpy as np

u = MDAnalysis.Universe(water_topology, water_trajectory)
ogrp = u.select_atoms("name ow")
hgrp = u.select_atoms("name hw")[::2]
dt = 0.2

vectors = mvec.get_vecarray(universe=u, agrp=ogrp, bgrp=hgrp)
vectors = mvec.norm_vecarray(vectors)[0]

ts_aniso, correl_anisolg2 = veccor.correlvec(vectors, refvec=[1,1,1], dt=dt, nlegendre=2, outfilename="anisolg2.dat", normed=True)
ts_isolg2, correl_isolg1 = veccor.isocorrelveclg2(vectors, dt=dt, outfilename="fast_isolg2.dat")
ts_iso, correl_iso = veccor.isocorrelvec(vectors, dt=dt, nlegendre=2, outfilename="slow_isolg2.dat")
