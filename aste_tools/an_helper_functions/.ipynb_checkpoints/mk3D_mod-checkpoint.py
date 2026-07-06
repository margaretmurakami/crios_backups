# path to this is /workspace/atnguyen/atn_tools/gcmfaces_mod on nansen
def mk3D_mod(arr_lessd, arr_3d):
    '''
    inputs
        arr_lessd - the array we want to make 3D
        arr_3d - the array in the shape we want
    outputs
        a - the modified array 
    '''
    import numpy as np
    # we will skip out on An's other definitions for gcmfaces and cells
    # Check if c is a double (numpy array)
    if isinstance(arr_3d, np.ndarray):
        nz = arr_3d.shape[0]
        full_size = np.array(arr_3d.shape)
        half_size = np.array(arr_lessd.shape)
        
        # If conditions for 2D->3D
        # go from 2D field to 3D field
        if len(half_size) == 2:
            tmp1 = arr_lessd.copy()
            n1 = arr_lessd.shape[0]
            n2 = arr_lessd.shape[1]
            #tmp1 = tmp1.flatten()
            #tmp1 = np.dot(arr_lessd.reshape(-1, 1), np.ones((1, arr_3d.shape[0])))
            #tmp1 = tmp1.reshape(arr_3d.shape[0],n1,n2)
            tmp1 = tmp1[np.newaxis,:,:] * np.ones((arr_3d.shape[0],1,1))
            a = tmp1

        # If conditions for 1D->3D
        elif len(half_size) == 1:
            tmp1 = arr_3d.copy()
            tmp2 = tmp1.shape
            n1 = tmp2[2]
            n2 = tmp2[1]
            #tmp1 = np.dot(np.ones((n1*n2,1)),arr_lessd[np.newaxis,:])
            #tmp1 = np.reshape(tmp1,(arr_3d.shape[0],n2,n1))
            tmp1 = np.ones((1,n2,n1)) * arr_lessd[:,np.newaxis,np.newaxis]
            a = tmp1
    return a