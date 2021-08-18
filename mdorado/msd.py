import MDAnalysis
import numpy as np
from scipy import signal

def unshift(universe, agrp, dimensionskey="xyz", cms=False):
    dimensionsdict = { "xyz": [0,1,2],
                      "x": [0],
                      "y": [1],
                      "z": [2],
                      "xy": [0,1],
                      "xz": [0,2],
                      "yz": [1,2],
                        }
    dimensions = dimensionsdict[dimensionskey]
    ulen = len(universe.trajectory)
    ndim = len(dimensions)
    boxangles = universe.coord.dimensions[3:]!=90
    if boxangles.any():
        raise ValueError("""
            Found at least one box angle which is not 90 degrees!
            """
            )
    if cms:
        na = len(agrp.atoms.residues)
        t0posarray = agrp.center(agrp.masses, compound='molecules')[:,dimensions]
    else:
        na = len(agrp)
        t0posarray = agrp.positions[:,dimensions]
    positions = np.zeros((ulen, na, ndim)) 
    vec = np.zeros((ulen, na, ndim)) 
    step = 1
    for ts in universe.trajectory[1:]:
        if cms:
            tposarray = agrp.center(agrp.masses, compound='molecules')[:,dimensions]
        else:
            tposarray = agrp.positions[:,dimensions]
        movearray = tposarray - t0posarray
        box = universe.coord.dimensions[dimensions]
        boxhalf = box * 0.5
        transarray = movearray.T
        for x in range(ndim):
            transarray[x, transarray[x]>boxhalf[x]] -= box[x] 
            transarray[x, transarray[x]<-boxhalf[x]] += box[x] 
        vec[step] = transarray.T
        positions[step] = positions[step-1] + transarray.T*0.1
        t0posarray = tposarray
        step+=1
    positions = np.moveaxis(positions,0,-1)
    return positions

def msd(positions, dt, outfilename="msd.dat"):
    na, ndim, ulen = positions.shape
    msd = np.zeros(ulen)
    for mol in positions:
        for x in range(ndim):
            xmsd = signal.correlate(mol[x], mol[x], method='auto', mode='full')
            twocorrel = 2 * xmsd[xmsd.size // 2:] 
            xsq = np.square(mol[x])
            s0 = 2 * np.sum(xsq)
            sm = s0 - np.cumsum(xsq)[:-1] - np.cumsum(xsq[1:][::-1])
            msd[0] += s0 - twocorrel[0]
            msd[1:] += sm - twocorrel[1:]
    factor =  1.0 / np.flip(np.arange(1, ulen+1))
    msd = factor * msd / na
    np.savetxt(outfilename, np.array([np.arange(ulen)*dt, msd]).T, fmt='%.10G')
