import numpy as np
import MDAnalysis
from MDAnalysis.analysis.distances import capped_distance

class Gofr:
    """
    alexandria.gofr.Gofr(universe, agrp, bgrp, rmax, rmin=0, bins=100,
        mode="site-site", outfilename="gofr.dat")

    Computes the average radial distribution function g(r). Between the
    sites in agrp and bgrp (mode="site-site"). If mode is "cms-cms" it
    first calculates the center-of-mass (cms) of atoms belonging to the 
    same molecule in those groups and then computes g(r) between those
    cms. The mode "site-cms" mixes both, and agrp is taken atom-wise 
    while for bgrp the cms are calculated.

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        agrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms A.

        bgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms B.

        rmax: int or float
            Upper boundary of the A-B distance used in g(r) in Angstrom.

        rmin: int or float, optional
            Lower boundary of the A-B distance used in g(r) in Angstrom.
            Default is 0.

        bins: int or sequence of scalars or str, optional
            Specifies the number of points between rmin and rmax. Is 
            directly used by numpy.histogram (see there for more info).
            The default is 100.

        mode: str, optional
            Mode for calculating g(r). Options are "site-site",
            "cms-cms", and "site-cms". If mode is set to "site-site",
            the average radial distribution function of all sites in 
            agrp to all sites in bgrp will be computed. The mode 
            "cms-cms" will first compute the center-of-mass of sites 
            belonging to the same molecule in agrp and bgrp, 
            respectively, and then determin g(r) between those centers
            of mass. The mode "site-cms" is a mix between the two, where
            every site in agrp is taken individualy but for bgrp the 
            center-of-mass of sites belonging to the same molecule is 
            computed firs. The default is "site-site".

        outfilename: str, optional
            The name of the output file. The default is "gofr.dat".

    Output
    ------
        The program creates a file with the distance r in Angstrom 
        (first column), the radial distribution function g(r) (second 
        column), the cumulative number of neighbors A in a sphere of 
        radius r around particle B N_A(r) (third column), and the 
        cumulative number of neighbors B in a sphere of radius r around 
        particle A N_B(r) (fourth column).


    Class Methods
    -------------
        rdat: ndarray, distance r (center of bins). 

        edges: ndarray, edges of the bins.

        hist: ndarray, values of g(r).

        annn: ndarray, neighbors A in a sphere of radius r around 
                particle B (N_A(r)).

        bnnn: ndarray, neighbors B in a sphere of radius r around 
                particle A (N_B(r)).

        avvol: float, average volume of the universe.

        na: int, number of particles A in agrp. If mode is "site-site" 
                or "site-cms", na is the number of sites in agrp. If 
                mode is "cms-cms", na is the number of molecules 
                (centers-of-mass) in agrp.

        nb: int, number of particles B in bgrp. If mode is "site-site" 
                nb is the number of sites in bgrp. If mode is "site-cms"
                or "cms-cms", nb is the number of molecules 
                (centers-of-mass) in bgrp.
    """

    def __init__(self, universe, agrp, bgrp, rmax, rmin=0, bins=100, mode="site-site", outfilename="gofr.dat"):
        #checking user input
        try:
            universe.trajectory
        except AttributeError:
            raise AttributeError("universe has no attribute 'trajectory'")
        try:
            universe.coord.dimensions
        except AttributeError:
            raise AttributeError("universe object has no attribute 'coord.dimensions'.")

        try:
            agrp.positions
        except AttributeError:
            raise AttributeError("agrp has no attribute 'positions'")

        try:
            bgrp.positions
        except AttributeError:
            raise AttributeError("bgrp has no attribute 'positions'")

        #initializing class methods
        self.u = universe
        self.agrp = agrp
        self.bgrp = bgrp
        self.rmax = float(rmax)
        self.rmin = float(rmin)
        self.bins = bins
        self.filename = str(outfilename)
        self.ulen = len(self.u.trajectory)
        #initializing histogram and edges
        self.hist, self.edges = np.histogram([], bins=self.bins, range=[self.rmin, self.rmax], density=False)
        self.hist = np.array(self.hist, dtype=np.float64)
        self.rdat = np.zeros(len(self.hist), dtype=np.float64)
        self.avvol = 0
        #query mode and call class methods accordingly
        if mode == "site-site":
            self._gofr()
        elif mode == "cms-cms":
            self._gofr_cms()
        elif mode == "site-cms":
            self._gofr_a_cms()
        else:
            raise ValueError("Gofr: mode has to be one of the following: site-site, cms-cms, site-cms")
        self._gatherdat()

    #site-site radial distribution function
    def _gofr(self):
        #na and nb are number of sites in agrp and bgrp
        self.na = int(self.agrp.__len__())
        self.nb = int(self.bgrp.__len__())
        #loop over all timesteps in universe
        for t in self.u.trajectory[::]:
            #box dimensions for periodic boundary conditions and average box volume
            dim = self.u.coord.dimensions
            vol = dim[0]*dim[1]*dim[2]
            self.avvol += vol
            #capped_distance seems to be faster without rmin
            pair, dab = capped_distance(self.agrp.positions, self.bgrp.positions, self.rmax, box=dim)
            #compute histogram from distances between sites, ihist represents numbers of entries, normalization follows later
            ihist, edges = np.histogram(dab, bins=self.bins, range=[self.rmin, self.rmax], density=False)
            self.hist += ihist*vol

    #cms-cms radial distribution function
    def _gofr_cms(self):
        #na and nb are number of residues in agrp and bgrp
        self.na = len(self.agrp.atoms.residues)
        self.nb = len(self.bgrp.atoms.residues)
        #loop over all timesteps in universe
        for t in self.u.trajectory[::]:
            #box dimensions for periodic boundary conditions and average box volume
            dim = self.u.coord.dimensions
            vol = dim[0]*dim[1]*dim[2]
            self.avvol += vol
            acms = self.agrp.center(self.agrp.masses, compound='residues')
            bcms = self.bgrp.center(self.bgrp.masses, compound='residues')
            #capped_distance seems to be faster without rmin
            pair, dab = capped_distance(acms, bcms, self.rmax, box=dim)
            #compute histogram from distances between cms, ihist represents numbers of entries, normalization follows later
            ihist, edges = np.histogram(dab, bins=self.bins, range=[self.rmin, self.rmax], density=False)
            self.hist += ihist*vol

    #site-cms radial distribution function
    def _gofr_a_cms(self):
        #na is number of sides in agrp
        self.na = int(self.agrp.__len__())
        #nb is number of residues in bgrp
        self.nb = len(self.bgrp.atoms.residues)
        #loop over all timesteps in universe
        for t in self.u.trajectory[::]:
            #box dimensions for periodic boundary conditions and average box volume (cuboid boxes)
            dim = self.u.coord.dimensions
            vol = dim[0]*dim[1]*dim[2]
            self.avvol += vol
            bcms = self.bgrp.center(self.bgrp.masses, compound='residues')
            #capped_distance seems to be faster without rmin
            pair, dab = capped_distance(self.agrp.positions, bcms, self.rmax, box=dim)
            #compute histogram from distances between site and cms, ihist represents numbers of entries, normalization follows later
            ihist, edges = np.histogram(dab, bins=self.bins, range=[self.rmin, self.rmax], density=False)
            self.hist += ihist*vol

    #normalization of histogram and output
    def _gatherdat(self):
        #normalization of hist using length of universe, number of A and B
        self.hist = self.hist / (self.ulen * self.na * self.nb)
        factor = 3.0/(4*np.pi)
        #average box volume
        self.avvol = self.avvol / self.ulen
        #number density of A and B to calculate annn and bnnn
        arhohist = (self.na / self.avvol) * self.hist
        brhohist = (self.nb / self.avvol) * self.hist
        self.annn = np.cumsum(arhohist)
        self.bnnn = np.cumsum(brhohist)
        for i in np.arange(len(self.hist)):
            #output r is in the middle of between two bin-edges
            self.rdat[i] = (self.edges[i] + self.edges[i+1]) * 0.5
            #normalizing hist using the sphere-shell volume
            self.hist[i] = factor*self.hist[i]/(self.edges[i+1]**3-self.edges[i]**3)
        #'%.10G': Floating point format. Uses uppercase exponential format if exponent is less than -4, decimal format otherwise.
        np.savetxt(self.filename, np.array([self.rdat, self.hist, self.annn, self.bnnn]).T, fmt='%.10G')
