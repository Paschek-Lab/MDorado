import numpy as np
from scipy import signal
from mdorado.correlations import correlate

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
    dimensionsdict = { "xyz": [0,1,2],
                      "x": [0],
                      "y": [1],
                      "z": [2],
                      "xy": [0,1],
                      "xz": [0,2],
                      "yz": [1,2],
                        }
    dimensions = dimensionsdict[dimensionskey]
    #initialize length of trajectory and number of dimensions
    ulen = len(universe.trajectory)
    ndim = len(dimensions)
    #verify that box is cuboid
    boxangles = universe.coord.dimensions[3:]!=90
    if boxangles.any():
        raise ValueError("""
            Found at least one box angle which is not 90 degrees!
            """
            )
    #initialize number of particles in agrp and positions at t=0, different if cms=True
    if cms:
        na = len(agrp.atoms.residues)
        t0posarray = agrp.center(agrp.masses, compound='molecules')[:,dimensions]
    else:
        na = len(agrp)
        t0posarray = agrp.positions[:,dimensions]
    #positions will contain the unwrapped trajectory of agrp
    positions = np.zeros((ulen, na, ndim)) 
    #loop over all timesteps in trajectory
    step = 1
    for ts in universe.trajectory[1:]:
        #current position
        if cms:
            tposarray = agrp.center(agrp.masses, compound='molecules')[:,dimensions]
        else:
            tposarray = agrp.positions[:,dimensions]
        #compute displacement vectors between two timesteps
        movearray = tposarray - t0posarray
        #reverse jumps due to periodic boundary conditions
        box = universe.coord.dimensions[dimensions]
        boxhalf = box * 0.5
        transarray = movearray.T
        for x in range(ndim):
            transarray[x, transarray[x]>boxhalf[x]] -= box[x] 
            transarray[x, transarray[x]<-boxhalf[x]] += box[x] 
        #update particle positions by adding displacement vector, unitconversion from A to nm
        positions[step] = positions[step-1] + transarray.T*0.1
        t0posarray = tposarray
        step+=1
    #rearrange trajectory array to shape (na, ndim, ulen)
    positions = np.moveaxis(positions,0,-1)
    return positions

def msd(positions, dt, outfilename="msd.dat"):
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

        outfilename: str, optional
            Name of the outputfile. Default is msd.dat.

    Output
    ------
        Write a file containing two columns, where the first column is
        the time delay t and the second column contains the average
        MSD(t).
    """
    #initialize atom number, number of dimensons and trajectory length
    na, ndim, ulen = positions.shape
    #initialize empty array to collect average MSD and factor for normalization
    msd = np.zeros(ulen)
    factor =  1.0 / np.flip(np.arange(1, ulen+1))
    #compute MSD for every particle and dimension seperately
    for mol in positions:
        for x in range(ndim):
            twocorrel = 2 * correlate(mol[x])
            xsq = np.square(mol[x])
            s0 = 2 * np.sum(xsq)
            sm = s0 - np.cumsum(xsq)[:-1] - np.cumsum(xsq[1:][::-1])
            msd[0] += s0*factor[0] - twocorrel[0]
            msd[1:] += sm*factor[1:] - twocorrel[1:]
    #normalize MSD to number of particles
    msd = msd / na
    np.savetxt(outfilename, np.array([np.arange(ulen)*dt, msd]).T, fmt='%.10G')
