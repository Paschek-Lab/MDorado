import numpy as np
from mdorado.vectors import norm_vecarray, get_vecarray
from mdorado.correlations import correlate




def build_combination(grp1,grp2, exclude_reverse=True, exclude_si=True):
    """
    mdorado.dipol_relax.build_combination(list1, list2, exclude_reverse=True, exclude_si=True)

    Helper function which builds all possible combinations between the 
    entries in list1 and list2.

    Parameters
    ----------
        grp1: list of str
            Atomtypes of equal subgroups.
            
        grp2: list of str
            Atomtypes of all atoms which could interact with this group.


        exclude_reverse: bool, optional
            Exclude reversed occurences from dict entries. 

        exclude_si: bool, optional
            Exclude occurences where both atomtypes are equal from dict entries.

    Returns
    -------
        comb_dict: dict 
        
        
        
    """
    
    comb=[]
    comb_dict=dict()
    cond=""
    
    # construct if conditional  
    if exclude_si:
        cond += "nucleus1 != nucleus2 "   
    if exclude_reverse:
        if len(cond)>1 :
            cond+= "and "     
        cond += "check not in comb "
    if len(cond) == 0:
        cond += "True"

    # print(f'Combination will be evaluated by: \n',cond)
    
    for nucleus1 in grp1:
        comb_dict[nucleus1]=[]
        for nucleus2 in grp2:
            check=[nucleus2,nucleus1]
            if eval(cond):
                comb.append([nucleus1,nucleus2])
                comb_dict[nucleus1].append([nucleus1,nucleus2])
                
    print("# of Interactions:", len(comb) )
    return comb_dict




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
    mdorado.dipol_relax.calc_ddrelax_intra(u, dt, comb_dict, coeff_dict, outfilename=None)

    Computes the intramolecular dipolar relaxation correlation function for a given set of
    combinations.  

    Parameters
    ----------
        universe: MD.Analysis.Universe
            The universe containing the trajectory.

        dt: int or float
            The difference in time between steps of the trajectory.

        comb_dict: dict
            contains the combination between different atom sites.
            Can be build by mdorado.dipol_relax.build_combination.
        
        coef_dict: dict
            contains the coefficient by which each correlation time 
            is multiplied if a reduction to subgroups is made.
        
        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing the timestep and corresponding value of the
            correlation function. If None (default), no file will be
            written.

    Returns
    -------
        timesteps, G2: ndarray, dict of ndarray
            Returns information about the timestep t and a dictionary
            containing the dipolar relaxation functions for every 
            comb_dict key. 
    """
    
    g2=dict()
    #select every key once as reference atomtype
    for ref_at in comb_dict.keys():
        cum_sum=0
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
            np.savetxt(f"{outfilename}_{ref_at}_intra.dat", np.array([timesteps, cum_sum]).T, fmt='%.10G')

    return timesteps,g2





def calc_ddrelax_inter(u,dt, comb_dict,coeff_dict, reference_residues,sub=True,maj=True,outfilename=None):
    """
    mdorado.dipol_relax.calc_ddrelax_inter(u,dt, comb_dict,coeff_dict, reference_residues,sub=True,maj=True,outfilename=None)

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
        
        coef_dict: dict
            contains the coefficient by which each correlation time 
            is multiplied if a reduction to subgroups is made.
            
        reference_residues: list of ints
            Contains the information of which resids should be calculated.
        
        sub: bool, optional
            Creats a file for each combination entry in comb_dict. Works only if 
            a filname with the outfilename  option is provided.
            
        maj: bool, optional
            Creats a file for each key in comb_dict. Works only if 
            a filname with the outfilename option is provided.
        
        outfilename: str, optional
            If specified an xy-file with the name str(outfilename)
            containing the timestep and corresponding value of the
            correlation function. If None (default), no file will be
            written.

    Returns
    -------
        timesteps, G2: ndarray, dict of ndarray
            Returns information about the timestep t and a dictionary
            containing the dipolar relaxation functions for every 
            comb_dict key. 
    """
    g2=dict()
    for ref in reference_residues:
        print('reference resiudes:',ref)
        g2[ref]=dict()
        maj_cum_sum= 0
        for key in comb_dict.keys():
            print("\t selected atomname:",key)
            sub_cum_sum=0
            for pair in  comb_dict[key]:
                
                # select only reference resid for agrp
                at1=u.select_atoms(f"name {pair[0]} and resid {ref}")
                if pair[0] == pair[1]:
                   at2=u.select_atoms(f"name {pair[1]} and not resid {ref} ")  
                else:
                    at2=u.select_atoms(f"name {pair[1]}")
                    
                at1_at2=get_vecarray(universe=u, agrp=at1, bgrp=at2, pbc=True)
                    
                timesteps,allcorrel= dipol_correl(at1_at2,dt)
                # collect correlation for each possible combinaion in dict entry
                sub_cum_sum+=allcorrel

            
            #multiply by occurences of atomgroup
            sub_cum_sum*= coeff_dict[key]
            g2[ref][key]=sub_cum_sum
            maj_cum_sum+= sub_cum_sum
            
            if outfilename is not None and sub:
                np.savetxt(f"{outfilename}_{ref}_{key}.dat", np.array([timesteps, sub_cum_sum]).T, fmt='%.10G')   
                      
        if outfilename is not None and maj:
            np.savetxt(f"{outfilename}_{ref}_inter.dat", np.array([timesteps, maj_cum_sum]).T, fmt='%.10G')

    return timesteps,g2

    
    
    