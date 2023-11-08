"""
mdorado.msd

Functions to help compute the mean square displacement.

Functions:
----------

    mdorado.msd.unwrap(universe, agrp, dimensionskey="xyz", cms=False)
        Unwraps the trajectory of an AtomGroup agrp (or its
        center-of-mass) and creates an array of shape N_a (number of
        atoms in agrp or number of cms), N_dim (number of dimensions),
        N_ts (number of timesteps), which can be used as an input to
        the msd function. Only works for boxes, where all angles are
        90°.

    mdorado.msd.msd(positions, dt, outfilename="msd.dat")
        Computes the mean-square-displacement for an array of particle
        positions of shape N_a, N_dim, N_ts (see unwrap function).
"""

import numpy as np
import mdorado.vectors as mvec
from mdorado.correlations import correlate

def unwrap(universe, agrp, dimensionskey="xyz", cms=False):
    """
    mdorado.msd.unwrap(universe, agrp, dimensionskey="xyz", cms=False)

    Unwraps the trajectory of an AtomGroup agrp (or its
    center-of-mass) and creates an array of shape N_a (number of
    atoms in agrp or number of cms), N_dim (number of dimensions),
    N_ts (number of timesteps), which can be used as an input to
    the msd function. Only works for boxes, where all angles are
    90°.

    Parameters
    ----------
        vecarray: MDAnalysis.Universe
            Universe containing the trajectory

        agrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms for which the trajectory
            should be unwrapped.

        dimensionskey: str, optional
            Dimension in which the trajectory is unwrapped. The
            keywords are:
                "xyz" for all dimensions
                "x", "y" or "z" for one of the three principle box axes
                "xy", "xz" or "yz" for a combination of two of the
                    three box axes.

        cms: bool, optional
            If cms=True the program computes the movement of the
            center-of-mass of atoms belonging to the same molecule in
            agrp. If, for example, agrp would contain all atoms of
            water molecules in the simulation this option allows for
            the calculation of the average center-of-mass MSD of these
            water molecules. If for the same case cms=False is chosen,
            the program calculates the MSD of all the various atoms
            individually, averaging over hydrogen as well as oxygen
            atoms. The default value is False.

    Returns
    -------
        ndarray
            An ndarray containing the unwrapped positions. The shape of
            the array is N_a, N_dim, N_steps, where N_a denotes the
            number of atoms in agrp (or the number of molecules if
            cms=True), N_dim denotes the number of dimensions according
            to the option dimensionskey, and N_steps is the number of
            timesteps in the universe.
    """

    #create dictionary and match the dimensionskey
    dimensionsdict = {"xyz": [0, 1, 2],
                      "x": [0],
                      "y": [1],
                      "z": [2],
                      "xy": [0, 1],
                      "xz": [0, 2],
                      "yz": [1, 2],
                      }
    dimensions = dimensionsdict[dimensionskey]
    #initialize length of trajectory and number of dimensions
    ulen = len(universe.trajectory)
    ndim = len(dimensions)
    #initialize number of particles in agrp and positions at t=0, different if cms=True
    if cms:
        na = len(agrp.atoms.residues)
        t0posarray = agrp.center(agrp.masses, compound='residues')[:, dimensions]
    else:
        na = len(agrp)
        t0posarray = agrp.positions[:, dimensions]
    #positions will contain the unwrapped trajectory of agrp
    positions = np.zeros((ulen, na, ndim))
    #loop over all timesteps in trajectory
    step = 1
    for ts in universe.trajectory[1:]:
        #current position
        if cms:
            tposarray = agrp.center(agrp.masses, compound='residues')[:, dimensions]
        else:
            tposarray = agrp.positions[:, dimensions]
        #compute displacement vectors between two timesteps
        movearray = tposarray - t0posarray
        #reverse jumps due to periodic boundary conditions
        box = universe.coord.dimensions[dimensions]
        movearray = mvec.pbc_vecarray(movearray, box)
        #update particle positions by adding displacement vector, unitconversion from A to nm
        positions[step] = positions[step - 1] + movearray * 0.1
        t0posarray = tposarray
        step += 1
    #rearrange trajectory array to shape (na, ndim, ulen)
    positions = np.moveaxis(positions, 0, -1)
    return positions

def msd(positions, dt, xyz=False, outfilename=False):
    """
    mdorado.msd.msd(positions, dt, outfilename="msd.dat")

    Computes the mean-square-displacement for an array of particle
    positions of shape N_a, N_dim, N_ts (see unwrap function) via
        MSD(t) = < | r(t) - r(0) |**2 >
    where r(t) denotes a positional vector of the unwrapped trajectory.
    Uses a Fast-Fourier-transform algorithm for long trajectories.


    Parameters
    ----------
        positions: ndarray
            Array of shape N_a, N_dim, N_steps containing an unwrapped
            trajectory, where N_a denotes the number of particles, N_dim
            denotes the number of dimensions, and N_steps denotes the
            number of steps (see mdorado.msd.unwrap).

        dt: int or float
            Difference in time between two configurations in the
            positions option.
        
        xyz: bool, optional
            If set to True, the function returns the MSD(t) for every dimension separately,
            averaged over all particles. The default value is False.

        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing t and MSD(t) will be written. If False (default),
            no file will be written.

    Returns
    -------
        timesteps, msd: ndarray, ndarray
            Returns two arrays, the first containing information about
            the timestep t and the second containing the averaged
            mean-square-displacement MSD(t)
    """
    #initialize atom number, number of dimensons and trajectory length
    na, ndim, ulen = positions.shape
    #initialize empty array to collect average MSD and factor for normalization
    msd_result = np.zeros((ulen, ndim))
    factor = 1.0 / np.flip(np.arange(1, ulen+1))
    #compute MSD for every particle and dimension seperately
    for mol in positions:
        for x in range(ndim):
            twocorrel = 2 * correlate(mol[x])
            xsq = np.square(mol[x])
            s0 = 2 * np.sum(xsq)
            sm = s0 - np.cumsum(xsq)[:-1] - np.cumsum(xsq[1:][::-1])
            msd_result[0,x] += s0*factor[0] - twocorrel[0]
            msd_result[1:,x] += sm*factor[1:] - twocorrel[1:]
    # average over all dimensions, if xyz=False
    if xyz == False:
        msd_result = np.sum(msd_result, axis=1)
    #normalize MSD to number of particles
    msd_result /= na
    timesteps = np.arange(ulen)*dt
    if outfilename:
        np.savetxt(str(outfilename), np.vstack((timesteps, msd_result.T)).T, fmt='%.10G')
    return timesteps, msd_result
