import numpy as np
from mdorado.vectors import norm_vecarray, get_vecarray, get_vectormatrix
from mdorado.correlations import correlate
from numpy import linalg as LA



def build_combination(grp1,grp2, exclude_selfinteraction=False, exclude_reverse=False,return_dict=True):
    """
    mdorado.dipol_relax.build_combination(  list1,
                                            list2,
                                            exclude_selfinteraction=False,
                                            exclude_reverse=False,
                                            return_dict=True
                                            )

    Helper function which builds all possible combinations between the 
    entries in list1 and list2.

    Parameters
    ----------
        grp1: list of str
            Atomtypes of equal subgroups.
            
        grp2: list of str
            Atomtypes of all atoms which could interact with this group.

        exclude_selfinteraction: bool, optional
            Exclude occurences where both atomtypes are equal.
            
        exclude_reverse: bool, optional
            Exclude occurences where the atomtypes are equal when reveresed.
        
        return_dict: bool, optional
            Retruns either a dict or a list of all specified combinations. 


    Returns
    -------
        comb_dict:  dict with combinations where the key is the first 
                    supplied atomtype list 
        comb: list with combination not grouped by atomtype 
        
        
    """
    
    comb=[]
    comb_dict=dict()
    cond=""
    
    # construct if conditional  
    if exclude_selfinteraction:
        cond += "nucleus1 != nucleus2 " 
    if exclude_reverse:
        cond += " and check not in comb"  
    if len(cond) == 0:
        cond += "True"
    
    for nucleus1 in grp1:
        comb_dict[nucleus1]=[]
        for nucleus2 in grp2:
            check=[nucleus2,nucleus1]
            if eval(cond):
                comb.append([nucleus1,nucleus2])
                comb_dict[nucleus1].append([nucleus1,nucleus2])
                
    print("# of Interactions:", len(comb) )
    if return_dict:
        return comb_dict
    else:
        return comb




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




def calc_ddrelax_intra(u,dt,comb_dict,coeff_dict,outfilename=None):
    """
    mdorado.dipol_relax.calc_ddrelax_intra( u,
                                            dt,
                                            comb_dict,
                                            coeff_dict,
                                            outfilename=None
                                            )

    Computes the intramolecular dipolar relaxation correlation function for a given set 
    of combinations.  

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        dt: int or float
            The difference in time between steps of the trajectory.

        comb_dict: dict
            contains the combination between different atom sites.
            Can be build by mdorado.dipol_relax.build_combination.
        
        coeff_dict: dict
            contains the coefficient by which each correlation time 
            is multiplied if a reduction to subgroups is made.
        
        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing the timestep and corresponding value of the
            correlation function. If None (default), no file will be
            written.

    Returns
    -------
        timesteps, g2: ndarray, dict of ndarray
            Returns information about the timestep t and a dictionary
            containing the dipolar relaxation functions for every 
            comb_dict key. 
    """
    
    g2=dict()
    ulen=u.trajectory.n_frames
    #select every key once as reference atomtype
    for ref_at in comb_dict.keys():
        cum_sum=np.zeros(ulen)
        for pair in comb_dict[ref_at]:
            at1=u.select_atoms(f"name {pair[0]}")
            at2=u.select_atoms(f"name {pair[1]}")            
            at1_at2= get_vecarray(universe=u, agrp=at1, bgrp=at2, pbc=True)
            n_resids=at1_at2.shape[0] 
            timesteps,allcorrel= dipol_correl(at1_at2,dt)
            cum_sum+=allcorrel/n_resids
        cum_sum*= coeff_dict[ref_at]
        g2[ref_at]=cum_sum
        
        if outfilename is not None:
            np.savetxt(f"{outfilename}_{ref_at}_total.dat"
                        , np.array([timesteps, cum_sum]).T, fmt='%.10G')

    return timesteps, g2





def calc_ddrelax_inter(u,dt, comb_dict,coeff_dict, ref_res,minor=False, major=True, outfilename=None):
    """
    mdorado.dipol_relax.calc_ddrelax_inter( u, 
                                            dt,
                                            comb_dict,
                                            coeff_dict,
                                            reference_residues,
                                            minor=False,
                                            major=True,
                                            outfilename=None
                                            )

    Computes the intermolecular dipolar relaxation correlation function for a given set of
    combinations for specified redids.  

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        dt: int or float
            The difference in time between steps of the trajectory.

        comb_dict: dict
            contains the combination between different atom sites.
            Can be build by mdorado.dipol_relax.build_combination.
        
        coeff_dict: dict
            contains the coefficient by which each correlation time 
            is multiplied if a reduction to subgroups is made.
            
        ref_res: int
            Set supplied resid as reference resid. 
        
        minor: bool, optional
            Creates a file for each value (atomtype combination) in comb_dict.
            Works only if a filname with the outfilename option is provided.
            
        major: bool, optional
            Creates a file for each key (reference atomtype) in comb_dict.
            Works only if a filname with the outfilename option is provided.
        
        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing the timestep and corresponding value of the
            correlation function. If None (default), no file will be
            written.

    Returns
    -------
        timesteps, g2: ndarray, dict of ndarray
            Returns information about the timestep t and a dictionary
            containing the dipolar relaxation functions for every 
            comb_dict key. 
    """
    g2=dict()

    print('reference resiudes:',ref_res)
    g2[ref_res]=dict()
    ulen=u.trajectory.n_frames
    maj_cum_sum= np.zeros(ulen)
    for key in comb_dict.keys():
        print("\t selected atomname:",key)
        sub_cum_sum = np.zeros(ulen)
        for pair in  comb_dict[key]:
            
            # select only reference resid for agrp
            at1=u.select_atoms(f"name {pair[0]} and resid {ref_res}")
            at2=u.select_atoms(f"name {pair[1]} and not resid {ref_res} ")  
            
            at1_at2=get_vecarray(universe=u, agrp=at1, bgrp=at2, pbc=True)
            
            timesteps,allcorrel= dipol_correl(at1_at2,dt)
            # collect correlation for each possible combinaion in dict entry
            sub_cum_sum+=allcorrel

        
        #multiply by occurences of atomgroup
        sub_cum_sum*= coeff_dict[key]
        g2[ref_res][key]=sub_cum_sum
        maj_cum_sum+= sub_cum_sum
        
        if outfilename is not None and minor:
            np.savetxt(f"{outfilename}_{key}_minor.dat", np.array([timesteps, sub_cum_sum]).T, fmt='%.10G')   
    if outfilename is not None and major:
        np.savetxt(f"{outfilename}_major.dat", np.array([timesteps, maj_cum_sum]).T, fmt='%.10G')

    return timesteps, g2

    
    

