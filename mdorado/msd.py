import numpy as np
from scipy import signal
from mdorado.correlations import correlate

def unwrap(universe, agrp, dimensionskey="xyz", cms=False):
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
