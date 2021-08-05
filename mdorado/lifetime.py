from multiprocessing import Pool
import numpy as np
import os
import MDAnalysis
from scipy import signal
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import calc_angles

#check for finished output files in case computation was restarted
def _check_files():
    #calclist will contain id of donors already computed
    calclist = []
    allarray = np.arange(_nh)
    for i in allarray:
        filename="ct_"+str(i)+".dat"
        if os.path.isfile(filename):
            #check for missing values if filename exists in case the writing process was interupted
            data = np.genfromtxt(filename, unpack=True, invalid_raise=False, missing_values="nan")
            if np.isnan(data).any()==False and len(data)==3 and len(data[0])==_ulen:
                calclist.append(i)
    #returns arry containing all donor ids not finished yet
    return np.delete(allarray, calclist, axis=0)

def _get_lt(donornr):
    #initialize data gathering arrays
    dhy = np.zeros((1, _ny))
    dxy = np.zeros((1, _ny))
    harray = np.zeros((_ulen, _ny), dtype=np.int8)
    capharray = np.zeros((_ulen, _ny), dtype=np.int8)
    i=0
    #loop over all timesteps
    for t in _univ.trajectory[::]:
        dim = _univ.coord.dimensions
        #H...Y distance computation for h(t)
        distance_array(_allh[donornr].position, _ally.positions, result=dhy, box=dim)
        #X...Y distance computation for H(t)
        distance_array(_allx[donornr].position, _ally.positions, result=dxy, box=dim)
        #consider distance cutoff
        indices = np.nonzero(dhy < _rchy)
        #angle computation only if distance is inside criterion
        radalpha = calc_angles(_allx[[donornr]*len(indices[0])].positions, _allh[[donornr]*len(indices[0])].positions, _ally[indices[1]].positions, box=dim)
        #consider angle cutoff
        alphaindices = np.nonzero(radalpha > _rad_cutoff)
        #gather data in timestep i of gathering-arrays
        harray[i][indices[1][alphaindices]] = 1
        capharray[i][dxy[0] < _rcxy] = 1
        i+=1
    #free-up memory for calculations ahead
    del radalpha, dhy, dxy, indices, alphaindices

    #if no hydrogen bond is present, the correlations are trivial
    if harray.any() == False:
        ct = np.zeros(_ulen)
        kint = np.zeros(_ulen)
    else:
        #transform array so that harray[i] is the i-th acceptor (not timestep as before)
        harray = harray.T
        #kill all acceptors not interacted with, they do not contribute to correlation but are considered in normalization
        indices = harray.sum(-1).nonzero()
        harray = harray[indices]
        #invharray = [1 - h(t)]
        invharray = np.negative(harray) + 1
        #if h(t)=0 for all t, H(t) does not matter as k_in(t)=0
        capharray = capharray.T[indices]
        #invcapharray = [1 - h(t)]*H(t)
        invhcapharray = np.multiply(invharray, capharray)
        #free-up memory for calculations ahead
        del invharray, capharray

        #initialize ct and kint as result gathering arrays
        ct = np.zeros(2*_ulen-1, dtype=np.int64)
        kint = np.zeros(2*_ulen-1)
        #calculate correlations for every donor-acceptor pair separately
        for i in np.arange(len(harray)):
            iharray = np.array(harray[i], dtype=np.int32)
            #ct = <h(0)*h(t)>
            ct+=signal.correlate(iharray, iharray, method='auto', mode='full')
            #kint = <hdot(0)*[1 - h(t)]*H(t)>; hdot = dh(t)/dt 
            kint += signal.correlate(invhcapharray[i], np.gradient(iharray, _dt), method='auto', mode='full') 
        #the correlation functions are only relevant for t>=0
        ct = ct[ct.size // 2:].astype(np.float64)
        kint = np.negative(kint[kint.size // 2:])

        #normalization of ct and kint
        factor = _norm/np.flip(np.arange(1, _ulen+1))
        ct = ct * factor
        kint = kint * factor
    #write output file
    outfilename = "ct_"+str(donornr)+".dat"
    tarray = np.arange(_ulen)*_dt
    np.savetxt(outfilename, np.array([tarray, ct, kint]).T, fmt='%.10G')

def calc_lifetime(universe, timestep, xgrp, hgrp, cutoff_hy, angle_cutoff, cutoff_xy, ygrp=None, nproc=1, check_memory=True):
    """
    alexandria.lifetime.calc_lifetime(universe, timestep xgrp, hgrp, 
    cutoff_hy, angle_cutoff, cutoff_xy, ygrp=None, nproc=1, 
    check_memory=True)

    Computes the correlation functions c(t)=<h(0)h(t)> and 
    k_in(t)= <hdot(0)*(1-h(t))*H(t)>; hdot = dh(t)/dt from trajectories
    and geometric interaction (e.g. hydrogen bond) criteria. The
    interaction (X-H...Y) is defined the H...Y distance and the angle
    XHY. These correlation functions are used to define lifetimes of the
    given interaction. 

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        timestep: int or float
            The difference in time between steps of the trajectory.

        xgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms X.

        hgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms H.

        cutoff_hy: int or float
            Criterion for the H...Y distance in Angstrom (upper
            boundary) to define h(t).

        angle_cutoff: int or float
            Criterion for the XHY angle in radian (lower boundary) to
            define h(t).

        cutoff_xy: int or float
            Criterion for the X...Y distance in Angstrom (upper
            boundary) to define H(t).

        ygrp: AtomGroup from MDAnalysis, optional.
            AtomGroup containing all atoms Y. If ygrp=None (default) 
            xgrp will be taken as acceptor group for the interaction
            X-H...X.

        nproc: int, optional
            Number of pool-workers (processes) used for parallel 
            computing. The default is 1.

        check_memory: bool, optional
           If True the program tries to make an educated guess as to if
           there is enough memory to hold the arrays during the 
           computation and stops the script if thats not the case.
           Setting it to False will skip this check. The default is
           True. 

    Output
    ------
        For every donor i in xgrp a file ct_i.dat will be created. The
        first column contains the timestep t in the same unit given in
        the option "timestep". The second column contains <h(0)h(t)>
        for that donor averaged over all acceptors. The third column
        contains <hdot(0)[1-h(t)]H(t)> for that donor, where 
        hdot=dh(t)/dt.
        Data of multiple donors can be averaged as long as the number
        of acceptors is constant for all donors.
    """

    #global variables needed in _get_lt function, not possible as class methods because class methods are not picklable (needed for parallelization)
    global _univ, _dt, _rad_cutoff, _allx, _allh, _ally, _ulen, _nh, _ny, _norm, _rchy, _rcxy 
    _univ = universe
    #check user input
    try:
        _ulen = len(_univ.trajectory)
    except AttributeError:
        raise AttributeError("universe has no attribute 'trajectory'")
    try: 
        _univ.coord.dimensions
    except AttributeError:
        raise AttributeError("universe object has no attribute 'coord.dimensions'.")
    _dt = float(timestep)
    
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

    _allx = xgrp
    _allh = hgrp
    _nh = int(_allh.__len__())
    if _nh != int(_allx.__len__()):  
        raise ValueError("hgrp and xgrp do not contain the same number of atoms")

    if ygrp is None:
        _ally = _allx
    else:
        try:
            ygrp.positions
        except AttributeError:
            raise AttributeError("ygrp has no attribute 'positions'")
        _ally = ygrp
    _ny = int(_ally.__len__())

    _rchy = float(cutoff_hy)
    _rcxy = float(cutoff_xy)
    _rad_cutoff = float(angle_cutoff)

    #X-H...X only has Nx-1 interacting pairs
    if _ally==_allx:
        _norm = 1.0/(_ny-1)
    else:
        _norm = 1.0/(_ny)
    
    #checks for already existing files in case program was restarted
    calclist = _check_files()
    #for large/long simulations the arrays may become to big for the available memory
    if check_memory == True:
        #use unix command "free" to query available memory
        availmem= float(os.popen('free -g').readlines()[1].split()[-1])
        approxmem = 4 * _ny * 10**(-9) * _ulen * nproc
        print("calc_lifetime: Approximate Memory Usage (Spike): ", approxmem, "GB")
        print("calc_lifetime: Available Memory: ", availmem, "GB")
        if approxmem > availmem: 
            raise Exception("""
calc_lifetime: The approximate memory usage exceeds the systems available memory! You can lower the expected memory usage by decreasing the amount of cores used in the calculation (nproc) or reducing the number of acceptors (ygrp). If you are positive that the memory usage is lower than expected you can set the option check_memory=False to circumvent this memory check. Beware that an overflow in memory may crash the program as well as the computation node.
                """)
    #multiprocessing via worker pool
    pool = Pool(processes=nproc)
    pool.map(_get_lt, calclist)
