import numpy as np
import MDAnalysis
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_angles

def hb_analyze(universe, xgrp, hgrp, rmax, ygrp=None, rmin=0, cosalphamin=-1, cosalphamax=1, bins=50, outfilename="hb_analyze.dat", ralphalist=False):
    """
    alexandria.hb_analyze.hb_analyze(universe, xgrp, hgrp, rmax, 
        ygrp=None, rmin=0, bins=50, outfilename="hb_analyze.dat",
        ralphalist=False)

    Creates a 2D probability density function used for analyzing 
    hydrogen bond interactions X-H...Y using the H...Y distance r and
    the cosine of the XHY angle alpha.  

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        xgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms X.

        hgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms H.

        rmax: int or float
            Upper boundary of the H-Y distance in Angstrom.

        ygrp: AtomGroup from MDAnalysis, optional.
            AtomGroup containing all atoms Y. If ygrp=None (default) 
            xgrp will be taken as acceptor group for the interaction
            X-H...X-

        rmin: int or float, optional
            Lower boundary of the H-Y distance in Angstrom. Default is
            0.
        
        cosalphamin: int or float, optional
            Lower boundary of the cosine of the XHY angle cos(alpha).
            The default is -1.

        cosalphamax: int or float, optional
            Upper boundary of the cosine of the XHY angle cos(alpha).
            The default is 1.

        bins: int or sequence of scalars or str, optional
            Specifies the number of points between rmin and rmax as 
            well as cosalphamin and cosalphamax. Is directly used by
            numpy.histogram2d (see there for more info). The default is
            50.

        outfilename: str, optional
            The name of the output file. The default is 
            "hb_analyze.dat".

        ralphalist: bool, optional
           Changes the output from the weighted probability density
           matrix to the list containing all the distances r and 
           corresponding coss(alpha) from which the probability density
           is calculated. The default is False. 
    Output
    ------
        The program creates a weighted twodimensional histogram. The 
        first axis represents the H...Y distance r and the second axis
        represents the cosine of the XHY angle cos(alpha). If 
        ralphalist=True the file contains the distances and 
        corresponding cosines of HY-pairs as a list: in the first column
        the distances are written in units of AA and the second column 
        indicates the cosine of the corresponding angle cos(alpha), both
        in the respective range rmin to rmax and cosalphamin to 
        cosalphamax.
    """
    #To do: 
    #implementing capped_distance for distance calculation 
    #implement dummy histogram to make more bins options work

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
        xgrp.positions
    except AttributeError:
        raise AttributeError("xgrp has no attribute 'positions'")

    try:
        xgrp[0].position
    except AttributeError:
        raise AttributeError("xgrp[0] has no attribute 'position'")

    try:
        hgrp.positions
    except AttributeError:
        raise AttributeError("hgrp has no attribute 'positions'")

    try:
        hgrp[0].position
    except AttributeError:
        raise AttributeError("hgrp[0] has no attribute 'position'")

    nh = int(hgrp.__len__())
    if nh != int(xgrp.__len__()):  
        raise ValueError("hgrp and xgrp do not contain the same number of atoms")

    if ygrp is None:
        ygrp = xgrp
    else:
        try:
            ygrp.positions
        except AttributeError:
            raise AttributeError("ygrp has no attribute 'positions'")
    ny = int(ygrp.__len__())

    rmin = float(rmin)
    rmax = float(rmax)
    cosalphamin = float(cosalphamin)
    cosalphamax = float(cosalphamax)

    bins = bins
    datacollect = open(outfilename, "w")
    #create r, cosalpha list for output
    if ralphalist==True:
        for t in universe.trajectory[::]:
            #box dimensions for periodic boundary condition
            dim = universe.coord.dimensions
            dhy = distance_array(hgrp.positions, ygrp.positions, box=dim)
            indices = np.nonzero((dhy < rmax) * (dhy > rmin))
            #only calculate angles where the distance is between the criteria
            radalpha = calc_angles(xgrp[indices[0]].positions, hgrp[indices[0]].positions, ygrp[indices[1]].positions, box=dim)
            #assignment dhy->radalpha works out for the flat dhy array (C-style/row-major)
            flatdhy = dhy[indices].flatten()
            flatcosalpha = np.cos(radalpha)
            #consider angle criteria
            indices = np.nonzero((flatcosalpha < cosalphamax) * (flatcosalpha > cosalphamin))
            flatcosalpha = flatcosalpha[indices]
            flatdhy = flatdhy[indices]
            for i in np.arange(len(flatdhy)):
                data_write = [flatdhy[i], flatcosalpha[i]]
                datacollect.write(" ".join(str(x) for x in data_write))        #output
                datacollect.write("\n")

    #create 2D matrix for output
    else:
        harray = np.zeros((bins, bins))
        for t in universe.trajectory[::]:
            #box dimensions for periodic boundary condition
            dim = universe.coord.dimensions
            dhy = distance_array(hgrp.positions, ygrp.positions, box=dim)
            indices = np.nonzero((dhy < rmax) * (dhy > rmin))
            #only calculate angles where the distance is between the criteria
            radalpha = calc_angles(xgrp[indices[0]].positions, hgrp[indices[0]].positions, ygrp[indices[1]].positions, box=dim)
            #assignment dhy->radalpha works out for the flat dhy array (C-style/row-major)
            flatdhy = dhy[indices].flatten()
            flatcosalpha = np.cos(radalpha)
            weights = (1.0/flatdhy)**2
            #angle criteria are considered in histogram range
            stepharray, xedges, yedges = np.histogram2d(flatdhy, flatcosalpha,  bins=bins, range=[[rmin, rmax], [cosalphamin, cosalphamax]], weights=weights, density=False)
            #histogram contains only the counts and can therefore be added up
            harray+=stepharray
        #compute probability density from the counts
        area = abs((xedges[0] - xedges[1]) * (yedges[0] - yedges[1]))
        allcount = np.sum(harray)
        norm = 1.0/(area * allcount)
        harray = np.log(np.multiply(harray, norm))
        for ilist in harray:
            datacollect.write(" ".join(str(x) for x in ilist))        #output
            datacollect.write("\n")
    datacollect.close()
