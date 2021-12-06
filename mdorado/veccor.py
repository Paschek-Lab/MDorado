import numpy as np
from mdorado.correlations import correlate
from scipy.special import eval_legendre

"""
mdorado.veccor

Functions to compute the reorientational correlation function
    R_i(t) = < P_i(0) P__i(t) >
using the i-th Legendre polynominal of cos(theta), where theta is the
angle between the molecular vector of interest and a fixed external 
axis.

Functions
---------
    mdorado.veccor.get_vec(universe, agrp, bgrp)
        Computes array of vectors for input to correlvec and 
        isocorrelvec.

    mdorado.veccor.get_normal_vec(universe, agrp, bgrp, cgrp)
        Computes array of normal vectors for input to correlvec and
        isocorrelvec.

    mdorado.veccor.correlvec(vecarray, refvec, dt, nlegendre, 
        outfilename=False, normed=True):
        Computes R_i(t) in the anisotropic case using a fixed external
        reference vector. 

    mdorado.veccor.isocorrelvec(vecarray, dt, nlegendre, 
        outfilename=False)
        Computes R_i(t) in the isotropic case. Slow compared to 
        mdorado.veccor.correlvec.

    mdorado.veccor.isocorrelveclg1(vecarray, dt, outfilename=False)
        Fast algorithm to compute R_1(t) in the isotropic case.

    mdorado.veccor.isocorrelveclg2(vecarray, dt, outfilename=False)
        Fast algorithm to compute R_2(t) in the isotropic case.
"""

def get_vec(universe, agrp, bgrp):
    """
    mdorado.veccor.get_vec(universe, agrp, bgrp)

    Computes given to AtomGroups agrp and bgrp, it computes the time 
    evolution of every vector bgrp[i]-agrp[i] for the whole trajectory.
    The resulting array can be used as an input for the correlvec or 
    isocorrelvec functions to compute a reorientational correlation 
    function R_i(t). Does not account for periodic boundary conditions,
    so only intramolecular vectors should be considered with 'repaired'
    trajectories, where molecules are not broken between two opposite
    sites of the box.

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        agrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms A.

        bgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms B.

    Returns
    -------
        ndarray
            Array containing the trajectory of every vector 
            bgrp[i]-agrp[i] of the shape N_vec (number of vectors), 
            N_ts (number of timesteps in the universe), N_dim (number
            of dimensions). The vectors are normalized before output. 
            Can be used as input for the correlvec and isocorrelvec 
            functions.
    """
    #checking user input
    try:
        universe.trajectory
    except AttributeError:
        raise AttributeError("universe has no attribute 'trajectory'")

    try:
        agrp.positions
    except AttributeError:
        raise AttributeError("agrp has no attribute 'positions'")

    try:
        bgrp.positions
    except AttributeError:
        raise AttributeError("bgrp has no attribute 'positions'")

    ulen = len(universe.trajectory)
    na = len(agrp)
    vecarray = np.zeros((ulen, na, 3))
    step = 0
    #compute vector for every timestep
    for ts in universe.trajectory:
        vecarray[step] = bgrp.positions - agrp.positions       #na = nb !
        step+=1
    #normalize vector
    norm = np.sqrt((vecarray*vecarray).sum(axis=-1))
    vecarray /= norm[:,:,np.newaxis]
    #reorder array for easier correlation
    vecarray = np.swapaxes(vecarray,0,1)
    return vecarray

def get_normal_vec(universe, agrp, bgrp, cgrp):
    """
    mdorado.veccor.get_normal_vec(universe, agrp, bgrp, cgrp)

    Computes given to AtomGroups agrp, bgrp and cgrp, it computes the
    time evolution of every normal vector 
        n = (bgrp[i]-agrp[i]) \cross (cgrp[i]-agrp[i]) 
    for the whole trajectory. The resulting array can be used as an
    input for the correlvec or isocorrelvec functions to compute a 
    reorientational correlation function R_i(t). Does not account for
    periodic boundary conditions, so only intramolecular vectors should
    be considered with 'repaired' trajectories, where molecules are not
    broken between two opposite sites of the box.

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        agrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms A used to define the
            plane containing the atoms A, B and C.

        bgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms B used to define the
            plane containing the atoms A, B and C.

        cgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms C used to define the
            plane containing the atoms A, B and C.

    Returns
    -------
        ndarray
            An array of shape N_vec (number of normal vectors), N_ts 
            (number of timesteps in the universe), N_dim (number of 
            dimensions) containing the time evolution of all normal
            vectors n = .bgrp[i]-agrp[i]) \cross (cgrp[i]-agrp[i]). The 
            vectors are normalized before output. Can be used as input 
            for the correlvec and isocorrelvec functions.
    """

    #checking user input
    try:
        universe.trajectory
    except AttributeError:
        raise AttributeError("universe has no attribute 'trajectory'")

    try: 
        agrp.positions
    except AttributeError:
        raise AttributeError("agrp has no attribute 'positions'")

    try: 
        bgrp.positions
    except AttributeError:
        raise AttributeError("bgrp has no attribute 'positions'")

    try: 
        cgrp.positions
    except AttributeError:
        raise AttributeError("cgrp has no attribute 'positions'")
    
    ulen = len(universe.trajectory)
    na = len(agrp)
    normvecarray = np.zeros((ulen, na, 3))
    step = 0
    #get the 2 vectors and compute normal vector for every timestep
    for ts in universe.trajectory:
        vec1array = bgrp.positions - agrp.positions       #na = nb ! 
        vec2array = cgrp.positions - agrp.positions       #na = nc !
        normvecarray[step] = np.cross(vec1array, vec2array)
        step+=1
    #normalize the normal vectors
    norm = np.sqrt((normvecarray*normvecarray).sum(axis=-1))
    normvecarray /= norm[:,:,np.newaxis]
    #reorder array for easier correlation
    normvecarray = np.swapaxes(normvecarray,0,1)
    return normvecarray

def correlvec(vecarray, refvec, dt, nlegendre, outfilename=False, normed=True):
    """
    mdorado.veccor.correlvec(vecarray, refvec, dt, nlegendre, 
        outfilename=False, normed=True)

    Computes the reorientational correlation function 
        R_i(t) = < P_i(0) P__i(t) >
    using the i-th Legendre polynominal P_i of cos(theta), where theta
    is the angle between the molecular vector of interest and a fixed
    external axis refvec.


    Parameters
    ----------
        vecarray: ndarray
            Array containing the trajectory of every vector of interest
            in the shape N_vec (number of vectors), N_ts (number of 
            timesteps in the universe), N_dim (number of dimensions).
            See functions get_vec and get_normal_vec for a way to
            obtain these functions

        refvec: ndarray or array-like
            Fixed external reference vector. Will be normalized
            internally.

        dt: int or float
            Timestep used in vecarray.

        nlegendre: int
            Specifies which Legendre polynomial i should be used to
            compute the respective reorientational correlation function
            R_i(t).

        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing t and R_i(t) will be written. If False (default),
            no file will be written. 
            
        normed: boolean, optional
            Specifies whether the function R_i(t) should be normalized
            or no. Default is True.

    Returns
    -------
        timesteps, allcorrel: ndarray, ndarray
            Returns two arrays, the first containing information about
            the timestep t and the second containing the averaged 
            function R_i(t).
    """

    na, ulen, ndim = vecarray.shape
    refvec = np.array(refvec)
    #normalize refvec
    norm = np.linalg.norm(refvec)
    refvec = refvec / norm
    allcorrel = np.zeros(ulen)
    #compute legendre polynomial and correlate vector-wise
    for mol in vecarray:
        legendre_poly = eval_legendre(nlegendre, np.inner(refvec, mol))
        allcorrel += correlate(legendre_poly)
    allcorrel = allcorrel / na
    #normalize correlation if necessary
    if normed == True:
        allcorrel /= allcorrel[0]
    timesteps = np.arange(ulen)*dt
    #save in textfile
    if outfilename:
        filename=str(outfilename)
        np.savetxt(filename, np.array([timesteps, allcorrel]).T, fmt='%.10G')
    return timesteps, allcorrel 

def isocorrelvec(vecarray, dt, nlegendre, outfilename=False):
    """
    mdorado.veccor.isocorrelvec(vecarray, dt, nlegendre, 
        outfilename=False)

    Computes the reorientational correlation function
        R_i(t) = < P_i(0) P__i(t) >
    using the i-th Legendre polynominal P_i of cos(theta), where theta 
    is the angle between the molecular vector of interest at time t and
    at time 0 (isotropic simplification).


    Parameters
    ----------
        vecarray: ndarray
            Array containing the trajectory of every vector of interest
            in the shape N_vec (number of vectors), N_ts (number of
            timesteps in the universe), N_dim (number of dimensions).
            See functions get_vec and get_normal_vec for a way to
            obtain these functions

        dt: int or float
            Timestep used in vecarray.

        nlegendre: int
            Specifies which Legendre polynomial i should be used to
            compute the respective reorientational correlation function
            R_i(t).

        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing t and R_i(t) will be written. If False (default),
            no file will be written.

    Returns
    -------
        timesteps, allcorrel: ndarray, ndarray
            Returns two arrays, the first containing information about
            the timestep t and the second containing the averaged
            function R_i(t).
    """

    na, ulen, ndim = vecarray.shape
    allcorrel = np.zeros(ulen)
    #compute legendre polynomial and correlate vector-wise
    for timediff in np.arange(ulen):
        avcorrel = 0
        for t0 in np.arange(ulen-timediff):
            vec1array = vecarray[:,t0] 
            vec2array = vecarray[:,t0+timediff] 
            legendre_poly = eval_legendre(nlegendre, np.sum(vec1array*vec2array, axis=1))
            avcorrel += np.mean(legendre_poly)
        allcorrel[timediff] = avcorrel / (ulen-timediff)
    timesteps = np.arange(ulen)*dt
    #save in textfile
    if outfilename:
        filename=str(outfilename)
        np.savetxt(filename, np.array([timesteps, allcorrel]).T, fmt='%.10G')
    return timesteps, allcorrel

def isocorrelveclg1(vecarray, dt, outfilename=False):
    """
    mdorado.veccor.isocorrelveclg1(vecarray, dt, outfilename=False)

    Computes the reorientational correlation function
        R_1(t) = < P_1(0) P__1(t) >
    using the first Legendre polynominal P_1 of cos(theta), where theta
    is the angle between the molecular vector of interest at time t and
    at time 0 (isotropic simplification). This function is faster than
    the normal isocorrelvec due to correlation via a Fast-Fourier-
    transform algorithm.


    Parameters
    ----------
        vecarray: ndarray
            Array containing the trajectory of every vector of interest
            in the shape N_vec (number of vectors), N_ts (number of
            timesteps in the universe), N_dim (number of dimensions).
            See functions get_vec and get_normal_vec for a way to
            obtain these functions

        dt: int or float
            Timestep used in vecarray.

        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing t and R_1(t) will be written. If False (default),
            no file will be written.

    Returns
    -------
        timesteps, allcorrel: ndarray, ndarray
            Returns two arrays, the first containing information about
            the timestep t and the second containing the averaged
            function R_1(t).
    """

    na, ulen, ndim = vecarray.shape
    allcorrel = np.zeros(ulen)
    #compute all parts of the expression vector-wise
    for mol in vecarray:
        #get x, y and z components of vector at all timesteps
        xarray = mol[...,0]
        yarray = mol[...,1]
        zarray = mol[...,2]
        xcorr = correlate(xarray)
        ycorr = correlate(yarray)
        zcorr = correlate(zarray)
        allcorrel += ( xcorr + ycorr + zcorr )
    #divide by number of vectors
    allcorrel =  allcorrel / na 
    timesteps = np.arange(ulen)*dt
    #save in textfile
    if outfilename:
        filename=str(outfilename)
        np.savetxt(filename, np.array([timesteps, allcorrel]).T, fmt='%.10G')
    return timesteps, allcorrel 

def isocorrelveclg2(vecarray, dt, outfilename=False):
    """
    mdorado.veccor.isocorrelveclg2(vecarray, dt, outfilename=False)

    Computes the reorientational correlation function
        R_2(t) = < P_2(0) P__2(t) >
    using the second Legendre polynominal P_2 of cos(theta), where theta
    is the angle between the molecular vector of interest at time t and
    at time 0 (isotropic simplification). This function is faster than
    the normal isocorrelvec due to correlation via a Fast-Fourier-
    transform algorithm.


    Parameters
    ----------
        vecarray: ndarray
            Array containing the trajectory of every vector of interest
            in the shape N_vec (number of vectors), N_ts (number of
            timesteps in the universe), N_dim (number of dimensions).
            See functions get_vec and get_normal_vec for a way to
            obtain these functions

        dt: int or float
            Timestep used in vecarray.

        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing t and R_2(t) will be written. If False (default),
            no file will be written.

    Returns
    -------
        timesteps, allcorrel: ndarray, ndarray
            Returns two arrays, the first containing information about
            the timestep t and the second containing the averaged
            function R_2(t).
    """

    na, ulen, ndim = vecarray.shape
    allcorrel = np.zeros(ulen)
    #compute all parts of the trinomial expression vector-wise
    for mol in vecarray:
        #get x, y and z components of vector at all timesteps
        xarray = mol[...,0]
        yarray = mol[...,1]
        zarray = mol[...,2]
        #parts of the trinomial expression (x**2+y**2+z**2+2xy+2xz+2yz)
        xsq = correlate(xarray*xarray)
        ysq = correlate(yarray*yarray)
        zsq = correlate(zarray*zarray)
        xy = correlate(xarray*yarray)
        xz = correlate(xarray*zarray)
        yz = correlate(yarray*zarray)
        allcorrel += ( xsq + ysq + zsq + 2*xy + 2*xz + 2*yz )
    #apply second legendre polynomial and divide by number of vectors
    allcorrel =  1.5 * allcorrel / na - 0.5
    timesteps = np.arange(ulen)*dt
    #save in textfile
    if outfilename:
        filename=str(outfilename)
        np.savetxt(filename, np.array([timesteps, allcorrel]).T, fmt='%.10G')
    return timesteps, allcorrel 
