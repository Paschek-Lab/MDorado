import numpy as np
from mdorado.vectors import norm_vecarray
from mdorado.correlations import correlate

def dipol_correl(vecarray, dt, outfilename=False):
    """
    mdorado.dipol_relax.dipol_correl(vecarray, dt, outfilename=False)

    Computes the dipolar relaxation correlation function for one or
    multiple vector trajectories.

    Parameters
    ----------
        vecarray: ndarray
            Array of shape N_vec (number of vectors), N_steps (number of
            timesteps in the trajectory), N_dim (number of dimension)
            containing all vectors of interest for the dipolar
            relaxation rate.

        dt: int or float
            The difference in time between steps of the trajectory.

        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing the timestep and corresponding value of the
            correlation function. If False (default), no file will be
            written.

    Returns
    -------
        timesteps, allcorrel: ndarray, ndarray
            Returns two arrays, the first containing information about
            the timestep t and the second containing the dipolar
            relaxation correlation function.
    """
    nvec, ulen = vecarray.shape[:2]
    vecarray, normarray = norm_vecarray(vecarray)
    indices = normarray.sum(-1).nonzero()
    normarray = normarray[indices]
    vecarray = vecarray[indices]
    nvec = len(normarray) #update number of vectors

    allcorrel = np.zeros(ulen)
    invnormcube_array = np.reciprocal(normarray)**3
    for mol in np.arange(nvec):
        xarray = vecarray[mol][..., 0]
        yarray = vecarray[mol][..., 1]
        zarray = vecarray[mol][..., 2]
        invnormcube = invnormcube_array[mol]

        xsq = correlate(xarray*xarray*invnormcube)
        ysq = correlate(yarray*yarray*invnormcube)
        zsq = correlate(zarray*zarray*invnormcube)
        xy = correlate(xarray*yarray*invnormcube)
        xz = correlate(xarray*zarray*invnormcube)
        yz = correlate(yarray*zarray*invnormcube)
        normcorrel = correlate(invnormcube)
        allcorrel += 1.5 * (xsq + ysq + zsq + 2*xy + 2*xz + 2*yz) - 0.5 * normcorrel

    timesteps = np.arange(ulen)*dt
    if outfilename:
        np.savetxt(str(outfilename), np.array([timesteps, allcorrel]).T, fmt='%.10G')
    return timesteps, allcorrel
