import numpy as np

"""
mdorado.vectors

Contains functions related to generate and manipulate vectors.

Functions:
---------

    mdorado.vectors.pbc_vecarray(vecarray, box)
        Applies periodic boundary condition to an array of vectors so
        that the resulting array satisfies the minimum image convention
        for cuboid simulation boxes (all angles are 90°).

    mdorado.vectors.norm_vecarray(vecarray)
        Given an array of vectors, normalizes each vector and returns the array of normalized vectors as well as the length of each vector.

    mdorado.vectors.vectormatrix(apos, bpos)
        Given two sets of particle positions [A_1, A_2, ..., A_n] and [B_1, B_2, ..., B_m] computes the vectors for all combinations A_iB_j.

    mdorado.vectors.get_vectormatrix(universe, agrp, bgrp, pbc=True)
        Uses mdorado.vectors.vectormatrix to compute a vector matrix containing all vectors between all combinations of particles A and B for every timestep.

    mdorado.vectors.get_vecarray(universe, agrp, bgrp, pbc=True)
        Computes given to AtomGroups agrp and bgrp, it computes the time
        evolution of every vector bgrp[i]-agrp[i] for the whole trajectory.
        Can account for periodic boundary conditions for cuboid boxes.

    mdorado.vectors.get_normal_vecarray(universe, agrp, bgrp, cgrp, pbc=True)
        Computes given to AtomGroups agrp, bgrp and cgrp, it computes the
        normal vectors
            n = (bgrp[i]-agrp[i]) \cross (cgrp[i]-agrp[i])
        for every timestep of the trajectory.
"""

def pbc_vecarray(vecarray, box):
    """
    mdorado.vectors.pbc_vecarray(vecarray, box)

    Applies periodic boundary condition to an array of vectors so that
    the resulting array satisfies the minimum image convention for
    cuboid simulation boxes (all angles are 90°).

    Parameters
    ----------
        vecarray: ndarray
            Array of shape N_vec (number of vectors), N_dim
            (dimensionality of the vectors) on which the periodic
            boundary condition will be applied.

        box: array-like
            One-dimensional array-like where the elements contain the
            length of the simulation box in the particular dimension.

    Returns
    -------
        ndarray
            Array of the same shape as vecarray containing the vectors
            corrected for the minimum image convention.
    """

    ndim = vecarray.shape[-1] 
    try: 
        boxhalf = box * 0.5
    except TypeError:
        box = np.array(box)
        boxhalf = box * 0.5
    transarray = 1 * vecarray.T
    #ndarray.T returns a view, meaning that in-place changes to
    #transarray will change vecarray. This is undesirable as the
    #function should return a new array without changing the input. 
    #"1 * ndarray.T" breaks that behavior and creates a new array.
    for dimension in range(ndim):
        transarray[dimension, transarray[dimension] > boxhalf[dimension]] -= box[dimension]
        transarray[dimension, transarray[dimension] < -boxhalf[dimension]] += box[dimension]
    return transarray.T

def norm_vecarray(vecarray):
    """
    mdorado.vectors.norm_vecarray(vecarray)

    Given an array of vectors, normalizes each vector and returns the array of normalized vectors as well as the length of each vector.

    Parameters
    ----------
        vecarray: ndarray
            Array of shape N_vec (number of vectors), N_dim (number of
            dimension) containing the vectors that will be normalized.

    Returns
    -------
        unitvecarray, norm: ndarray, ndarray
            Returns two arrays, the first (unitvecarray) is of the same
            shape as the input array and contains the normalized vectors
            and the second (norm) is of the shape (N_vec,) and contains
            the length of the corresponding original vector.
    """

    norm = np.sqrt((vecarray*vecarray).sum(axis=-1))
    unitvecarray = vecarray / norm[..., np.newaxis]
    return unitvecarray, norm

def vectormatrix(apos, bpos):
    """
    mdorado.vectors.vectormatrix(apos, bpos)

    Given two sets of particle positions [A_1, A_2, ..., A_n] and [B_1, B_2, ..., B_m] computes the vectors for all combinations A_iB_j. 

    Parameters
    ----------
        apos: ndarray
            Array of shape (n, 3) containing the position vectors of all particles A.

        bpos: ndarray
            Array of shape (m, 3) containing the position vectors of all particles B.

    Returns
    -------
        ndarray
            Returns an array of shape (n, m, 3) containing all vectors 
            AB[i, j] = bpos[j] - apos[i].
    """

    lenapos = len(apos)
    lenbpos = len(bpos)
    resarray = np.zeros((lenapos, lenbpos, 3), dtype=np.float32)
    for dimension in range(3):
        xpos1 = np.reshape(apos[..., dimension], (lenapos, 1))
        xpos2 = np.reshape(bpos[..., dimension], (lenbpos, 1))
        #Make sure that AB[i, j] = B_j - A_i
        abvecmatrix = xpos2.T - xpos1
        resarray[..., dimension] = abvecmatrix
    return resarray

def get_vectormatrix(universe, agrp, bgrp, pbc=True):
    """
    mdorado.vectors.get_vectormatrix(universe, agrp, bgrp, pbc=True)
    
    Uses mdorado.vectors.vectormatrix to compute a vector matrix containing all vectors between all combinations of particles A and B for every timestep in a universe.

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        agrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms A.

        bgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms B.

        pbc: bool, optional
            Specifies whether periodic boundary conditions should be
            applied to find the shortest vector from A to an image of B.
            Calls the mdorado.vectors.pbc_vecarray function. Only works
            for cuboid boxes (all angles are 90°).

    Returns
    -------
        ndarray
            Array of the shape (N_steps, N_A, N_B, 3), where N_steps is
            the number of timesteps in the universe, N_A the number of
            particles A, N_B the number of particles B, and the last
            axis refers to the three directions in space x, y and z. For
            example, the array element AB[i, j, k, 2] refers to the z-
            component of the vector pointing from A_j to B_k at the i-th
            timestep of the simulation.
    """

    ulen = len(universe.trajectory)
    vecmatrix = np.zeros((ulen, len(agrp), len(bgrp), 3), dtype=np.float32)
    step = 0
    for ts in universe.trajectory:
        apos = agrp.positions
        bpos = bgrp.positions
        crossarray = vectormatrix(apos, bpos)
        if pbc:
            box = universe.coord.dimensions
            vecmatrix[step] = pbc_vecarray(crossarray, box)
        else:
            vecmatrix[step] = crossarray
        step += 1
    vecmatrix = np.moveaxis(vecmatrix, 0, 2)
    return vecmatrix

def get_vecarray(universe, agrp, bgrp, pbc=True):
    """
    mdorado.vectors.get_vecarray(universe, agrp, bgrp, pbc=True)

    Given two AtomGroups agrp and bgrp, it computes the time evolution
    of every vector bgrp[i]-agrp[i] for the whole trajectory. Can 
    account for periodic boundary conditions for cuboid boxes.

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        agrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms A.

        bgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms B.

        pbc: bool, optional
            Specifies whether periodic boundary conditions should be
            applied to find the shortest vector from A to an image of B.
            Calls the mdorado.vectors.pbc_vecarray function. Only works
            for cuboid boxes (all angles are 90°). Default is True.
            
    Returns
    -------
        ndarray
            Array containing the trajectory of every vector
            bgrp[i]-agrp[i] of the shape N_vec (number of vectors),
            N_ts (number of timesteps in the universe), N_dim (number
            of dimensions).
    """

    ulen = len(universe.trajectory)
    vecarray = np.zeros((ulen, len(agrp), 3), dtype=np.float32)
    step = 0
    for ts in universe.trajectory:
        try:
            diffarray = bgrp.positions - agrp.positions
        except ValueError:
            print("Error: agrp and bgrp have to contain the same number of atoms!")
            raise
        if pbc:
            box = universe.coord.dimensions
            vecarray[step] = pbc_vecarray(diffarray, box)
        else:
            vecarray[step] = diffarray
        step += 1
    vecarray = np.swapaxes(vecarray, 0, 1)
    return vecarray

def get_normal_vecarray(universe, agrp, bgrp, cgrp, pbc=True):
    """
    mdorado.vectors.get_normal_vecarray(universe, agrp, bgrp, cgrp, pbc=True)

    Computes given to AtomGroups agrp, bgrp and cgrp, it computes the
    normal vectors
        n = (bgrp[i]-agrp[i]) \cross (cgrp[i]-agrp[i])
    for every timestep of the trajectory.

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        agrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms A used to define the
            plane containing the atoms A_i, B_i and C_i.

        bgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms B used to define the
            plane containing the atoms A_i, B_i and C_i.

        cgrp: AtomGroup from MDAnalysis
            AtomGroup containing all atoms C used to define the
            plane containing the atoms A_i, B_i and C_i.

        pbc: bool, optional
            Specifies whether periodic boundary conditions should be
            applied to find the shortest vector from A to an image of B
            as well as from A to an image of C. Calls the 
            mdorado.vectors.pbc_vecarray function. Only works for cuboid
            boxes (all angles are 90°).

    Returns
    -------
        ndarray
            An array of shape N_vec (number of normal vectors), N_ts
            (number of timesteps in the universe), N_dim (number of
            dimensions) containing the time evolution of all normal
            vectors n = bgrp[i]-agrp[i]) \cross (cgrp[i]-agrp[i]).
    """

    ulen = len(universe.trajectory)
    normalvecarray = np.zeros((ulen, len(agrp), 3))
    step = 0
    #get the 2 vectors and compute normal vector for every timestep
    for ts in universe.trajectory:
        try:
            vec1array = bgrp.positions - agrp.positions
            vec2array = cgrp.positions - agrp.positions
        except ValueError:
            print("Error: agrp, bgrp and cgrp have to contain the same number of atoms!")
            raise
        if pbc:
            box = universe.coord.dimensions
            vec1array = pbc_vecarray(vec1array, box) 
            vec2array = pbc_vecarray(vec2array, box) 
        normalvecarray[step] = np.cross(vec1array, vec2array)
        step+=1
    #reorder array for easier correlation
    normalvecarray = np.swapaxes(normalvecarray,0,1)
    return normalvecarray
