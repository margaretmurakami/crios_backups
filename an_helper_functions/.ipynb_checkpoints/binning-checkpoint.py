import numpy as np
import xarray as xr

'''
This file will contain several helper functions for the binning we need to do

'''

# let's create a binning function with numpy
def bin_array(arr, bin_edges):
    '''
    Create a binned array based on edges of given bins

    Inputs:
        arr: the array we want to bin
        bin_edges: the edges of the bins we want to categorize

    Outputs:
        bin_indices_3d: the 3D array of indices in bin_edges where we find our values

    example:
        arr = [0.5,1.7,3.1]
        bin_edges = [0,1,2,3,4]

        binned = bin_array(arr,bin_edges)
            output: array([0, 1, 3])
    '''
    
    flattened_arr = arr.flatten()
    bin_indices = np.digitize(flattened_arr, bin_edges)
    bin_indices_3d = bin_indices.reshape(arr.shape)
    
    return bin_indices_3d-1

# create a TS mesh from a dataset, no longer use this function but I did in the beginning
def create_mesh(snap,ds,nS,nT,npoints,attr,mskBasin,iB,dT,dS):
    '''
    Inputs:
        snap: +1 or -1 in accordance with mitgcm text, + is time-averaged, - is a snap set
        ds: dataset with values we want, salinity, temperature
        nS: binsSLT_edges.shape[0]-1
        nT: binsTH_edges.shape[0]-1
        npoints: number of points in the basin we want
        attr: the attribute we want to create the mesh for
        iB: the index in mskBasin

    Outputs:
        testmesh: mesh of shape times,nS,nT,npoints of the binned attribute values at each time step
            this is in units of attr/deg C/PSU
    '''
    if snap < 0:
        testmesh = np.zeros((len(ds.iteration.values),nS, nT, npoints))     #ntimes x nS x nT x points in basin
        times = ds.iteration.values

        tn = 0
        for t in ds.iteration.values:
            dsx = ds.sel(iteration = t)
            if len(dsx[attr].values.shape) == 3:
                # 3D, bin in 3D
                thisvol = dsx[attr].values[:,mskBasin == iB]                 # all depths, mskBasin points 
                thissalt = dsx.salinity_binned.values[:,mskBasin == iB]
                thistemp = dsx.theta_binned.values[:,mskBasin == iB]
            elif len(dsx[attr].values.shape) == 2:
                # 2D field, only use top layer of binned salt and temp
                thisvol = dsx[attr].values[mskBasin == iB]                    # these should just be at the surface
                thissalt = dsx.salinity_binned.values[0,mskBasin == iB]       # these should just be at the surface
                thistemp = dsx.theta_binned.values[0,mskBasin == iB ]         # these should just be at the surface
    
            # trim the fat (nan values)
            thisvol = np.where(np.isnan(thisvol), 0, thisvol)
            thissalt = np.where(np.isnan(thissalt), -1, thissalt)  # Replace NaN with -1
            thistemp = np.where(np.isnan(thistemp), -1, thistemp)  # as above, indexing should not matter because this should be 0 volume
    
            # create the mesh
            meshx = np.zeros((nS, nT, npoints))
        
            saltflat = thissalt.flatten()
            tempflat = thistemp.flatten()
        
            # create local timed mesh
            np.add.at(meshx, (thissalt.astype(int), thistemp.astype(int), np.arange(0,npoints,1)), thisvol[...])  # this should work to add at bins
            meshx /= dT   # m^3/deg C
            meshx /= dS   # m^3/deg C/PSU
        
            # add to big mesh
            testmesh[tn,:,:,:] = meshx
        
            # delete for memory
            del meshx
            
            tn += 1
    else:
        testmesh = np.zeros((1,nS, nT, npoints))
        t = ds.iteration.values[1]   # assuming there are only two iterations in the dataset
        tn = 0
        dsx = ds.sel(iteration = t)
        if len(dsx[attr].values.shape) == 3:
            # 3D, bin in 3D
            thisvol = dsx[attr].values[:,mskBasin == iB]                 # all depths, mskBasin points 
            thissalt = dsx.salinity_binned.values[:,mskBasin == iB]
            thistemp = dsx.theta_binned.values[:,mskBasin == iB]
        elif len(dsx[attr].values.shape) == 2:
            # 2D field, only use top layer of binned salt and temp
            thisvol = dsx[attr].values[mskBasin == iB]                    # these should just be at the surface
            thissalt = dsx.salinity_binned.values[0,mskBasin == iB]       # these should just be at the surface
            thistemp = dsx.theta_binned.values[0,mskBasin == iB ]         # these should just be at the surface

        # trim the fat (nan values)
        thisvol = np.where(np.isnan(thisvol), 0, thisvol)
        thissalt = np.where(np.isnan(thissalt), -1, thissalt)  # Replace NaN with -1
        thistemp = np.where(np.isnan(thistemp), -1, thistemp)  # as above, indexing should not matter because this should be 0 volume

        # create the mesh
        meshx = np.zeros((nS, nT, npoints))
    
        saltflat = thissalt.flatten()
        tempflat = thistemp.flatten()
    
        # create local timed mesh
        np.add.at(meshx, (thissalt.astype(int), thistemp.astype(int), np.arange(0,npoints,1)), thisvol[...])  # this should work to add at bins
        meshx /= dT   # m^3/deg C
        meshx /= dS   # m^3/deg C/PSU
    
        # add to big mesh
        testmesh[tn,:,:,:] = meshx
    
        # delete for memory
        del meshx
        
        tn += 1

    
    return testmesh

def create_TS_mesh(tsstr,nS,nT,npoints, binned_salinity, binned_theta, attr,idxs,dT,dS):
    '''
    Inputs:
        nS: binsSLT_edges.shape[0]-1
        nT: binsTH_edges.shape[0]-1
        binned_salinity: the array of shape nz, ny, nx of the indices of salinity in the salt bins
        binned_theta: same as above but for theta
        attr: the attribute we want to bin, ie advection, diffusion etc.
        idxs: np.where(mymsk == iB) or whatever indices in mskBasin we are looking at

    Outputs:
        returns an nS by nT shaped array with the summed values within the attr (like volume)
    '''
    
    mesh = np.zeros((len(tsstr),nS, nT, npoints))
    tn = 0
    for t in range(len(tsstr)):
        if len(attr.shape) == 4:
            # time x nz x ny x nx
            thisvol = attr[t][:,idxs[0],idxs[1]]
            thissalt = binned_salinity[t][:,idxs[0],idxs[1]]
            thistemp = binned_theta[t][:,idxs[0],idxs[1]]
        elif len(attr.shape) == 3:
            # time x ny x nx
            thisvol = attr[t][idxs[0],idxs[1]]
            thissalt = binned_salinity[t][idxs[0],idxs[1]]
            thistemp = binned_theta[t][idxs[0],idxs[1]]
            
        # trim the nan values
        thisvol = np.where(np.isnan(thisvol), 0, thisvol)
        thissalt = np.where(np.isnan(thissalt), -1, thissalt)  # Replace NaN with -1
        thistemp = np.where(np.isnan(thistemp), -1, thistemp)
        
        # create the mesh
        meshx = np.zeros((nS,nT,npoints))
        
        # create local timed mesh
        np.add.at(meshx, (thissalt.astype(int), thistemp.astype(int), np.arange(0,npoints,1)), thisvol[...])  # this should work to add at bins
        meshx /= dT   # m^3/deg C
        meshx /= dS   # m^3/deg C/PSU
        
        # add to big mesh
        mesh[tn,:,:,:] = meshx
        del meshx

        tn += 1
    return mesh


# function to create G vectors from tendency terms
# to create Jx or Jy from this we just do d(G_S)/dT or d(G_T)/dS, respectively
# create the mesh of vectors for GS and GT
def calc_G_term(attr,binned_theta,binned_salinity,nT,nS,binwidthS,binwidthT):
    '''
    Creates the G_S and G_T terms provided either attr, the S tend, or attr, the T tend
    
    Inputs:
        attr: for one time step, the tendency in terms of PSU.m^3/s or degC.m^3/s (shape nz,ny,nx)
        binned_theta,binned_salinity: the indices of the salt and temp bins for each nz,ny,nx point at that tstep
        nT, nS: number of cell centers
        binwidthT, binwidthS: bin widths in T-S space for comparison

    Outputs:
        an array of shape nT, nS of the G term values for a given basin or set of basins
    '''
    # initialize the TS mesh
    distr_attr = np.full((nT,nS),0.0)

    # initialize tiles for binwidths
    binwidthsS_tile = np.tile(binwidthS, (112, 1)).T
    binwidthsT_tile = np.tile(binwidthT, (112, 1))

    # get the indices where not nan so that we don't have to loop, hopefully faster
    indices = np.where(~np.isnan(binned_theta))
    if len(indices) == 2:
        y,x = indices[0],indices[1]
    else:
        z,y,x = indices[0],indices[1],indices[2]

    # loop over the z,y,x indices, grab the values from attr, and add them to the mesh
    # 2D case (ie gate)
    if len(indices) == 2:
        for i,j in zip(y,x):
            distr_attr[int(binned_salinity[i,j]),int(binned_theta[i,j])] += attr[i,j]

    # 3D case (ie Basin)
    elif len(indices) == 3:
        for i,j,k in zip(z,y,x):
            distr_attr[int(binned_salinity[i,j,k]),int(binned_theta[i,j,k])] += attr[i,j,k]    # degC.m^3/s OR PSU.m^3/s

    # divide by binwidthsS and binwidthsT tiles
    # heat tendency becomes Sv/PSU, salt tendency becomes Sv/degC
    #distr_attr  = distr_attr/binwidthsT_tile[:,:]/binwidthsS_tile[:,:]    # m^3/s/PSU OR m^3/s/degC
    distr_attr *= 1e-6                                                    # Sv/PSU or Sv/degC
    distr_attr[distr_attr == 0 ] = np.nan

    return distr_attr