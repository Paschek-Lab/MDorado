import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as col

lowcolor = '#ffffff'
midcolor1 = '#6090f0'
midcolor2 = '#30f050'
midcolor3 = '#f0f000'
midcolor4 = '#f06000'
highcolor = '#b02000'
cmapown = col.LinearSegmentedColormap.from_list('own',[lowcolor,midcolor1,midcolor2,midcolor3,midcolor4,highcolor])
cmapown.set_over('#9e1c00')

rmin=1.5
rmax=5
cosalphamin=-1
cosalphamax=1

fig, ax = plt.subplots()
histo_matrix = np.loadtxt("hb_analyze.dat")
levels = ticker.MaxNLocator(nbins=60).tick_values(-3, 2)
cax = ax.contourf(histo_matrix, extent=(cosalphamin, cosalphamax, rmin, rmax), levels=levels, extend='both', cmap=cmapown)
plt.xlabel('$\\cos(\\alpha)$')
plt.ylabel('$r$ / \\AA')
plt.axis([cosalphamin, cosalphamax, rmin, rmax])
cbar = fig.colorbar(cax, ticks=[-3, -2, -1, 0, 1, 2])
cbar.ax.set_ylabel('$\\log[W(\\cos(\\alpha), r)]$')
plt.tight_layout()
plt.savefig("histo.pdf")
plt.clf()
