def calc_UV_conv_1face(U,V):
    import numpy as np
    # Check size
    sv = V.shape
    sz = U.shape

    # I think these are wrong
    if len(sz) == 2:

        tmp = np.full((1,sz[0],sz[1]),np.nan)
        sz = tmp.shape

    if len(sz) == 3:

        up = np.full((sz[0], sz[1], sz[2]+1), np.nan)
        up[:sz[0],:sz[1],:sz[2]] = U

        vp = np.full((sz[0],sz[1]+1,sz[2]), np.nan)
        vp[:sz[0],:sz[1],:sz[2]] = V

        uc = np.full((sz[0], sz[1], sz[2]),np.nan)
        vc = np.full((sz[0], sz[1], sz[2]),np.nan)
        
        uc = up[:, :, :-1] - up[:, :, 1:]
        vc = vp[:, :-1, :] - vp[:, 1:, :]
        
        fldOut = uc + vc
        
    return fldOut