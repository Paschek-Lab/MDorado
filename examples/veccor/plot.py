import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

fig, ax = plt.subplots(1)


t, vec = np.loadtxt("slow_isolg2.dat", unpack=True)
ax.plot(t,vec, color="k", label="isotropic (slow)")

t, vec = np.loadtxt("fast_isolg2.dat", unpack=True)
ax.plot(t,vec, color="orange",ls="dashed", label="isotropic (fast)")

t, vec = np.loadtxt("anisolg2.dat", unpack=True)
ax.plot(t,vec, color="royalblue", label="anisotropic")

ax.set_ylim(0.01,1)
ax.set_xlim(0.2,100)
plt.xscale("log")
plt.yscale("log")
plt.xlabel('$t$ / ps')
plt.ylabel('$C(t)$')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('vgl.pdf')#png, pdf, ps, eps and svg
plt.clf()
