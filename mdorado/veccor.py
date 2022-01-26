"""
mdorado.veccor

Functions to compute the reorientational correlation function
    R_i(t) = < P_i(0) P__i(t) >
using the i-th Legendre polynominal of cos(theta), where theta is the
angle between the molecular vector of interest and a fixed external 
axis.

Functions
---------
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

import numpy as np
from mdorado.correlations import correlate
from scipy.special import eval_legendre

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
            See module mdorado.vectors for ways to obtain such arrays.

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
            See module mdorado.vectors for ways to obtain such arrays.

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
            See module mdorado.vectors for ways to obtain such arrays.

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
            See module mdorado.vectors for ways to obtain such arrays.

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
