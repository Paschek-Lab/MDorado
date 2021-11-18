import numpy as np
from mdorado.correlations import correlate
from scipy.special import eval_legendre

def get_normal_vec(universe, agrp, bgrp, cgrp):
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

def get_vec(universe, agrp, bgrp):  
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

def isocorrelvec(vecarray, dt, nlegendre, outfilename=False):
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

def correlvec(vecarray, refvec, dt, nlegendre, outfilename=False, normed=True):
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
