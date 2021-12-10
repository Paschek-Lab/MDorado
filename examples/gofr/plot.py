import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

fig, ax = plt.subplots(1)
r, gofr = np.loadtxt('h_o.dat', unpack=True, usecols=(0,1))
gofrcms = np.loadtxt('cms_cms.dat', unpack=True, usecols=(1))
gofracms = np.loadtxt('h_cms.dat', unpack=True, usecols=(1))

ax.plot(r,gofr, color='firebrick', label="H$\cdots$O")
ax.plot(r,gofrcms, color='royalblue', label="cms$\cdots$cms")
ax.plot(r,gofracms, color='orange', label="H$\cdots$cms")

ax.set_xlim(1.0,6)
ax.set_ylim(0,4)
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
plt.xlabel(' $r$ / \\AA')
plt.ylabel('$g(r)$')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('gofr.pdf')#png, pdf, ps, eps and svg
plt.clf()
