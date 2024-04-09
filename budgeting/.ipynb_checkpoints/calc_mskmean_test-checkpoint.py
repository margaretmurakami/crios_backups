# original function is in /home/atnguyen/matlab/atn_tools on sverdrup
def calc_mskmean_T_mod(fldIn, mask, RAC, fldType="intensive"):
    """
    Computes average over a region (mask) of fldIn (or its fields recursively).
    If fldType is 'intensive' (default) then fldIn is multiplied by RAC.

    inputs
        fldIn dictionary (tend is kg, hconv is s^-1, zconv is s^-1)
        h array (2D)
        grid area RAC array (2D)
    outputs
        tmp dictionary with tend, hconv, zconv
    """
    import numpy as np
    
    # If fldIn is a dictionary
    if isinstance(fldIn, dict):
        fldOut = {}
        for key, value in fldIn.items():
            if isinstance(value, (float, int, np.ndarray)):
                tmp2, area = calc_mskmean_T_mod(value, mask, RAC, fldType)
                fldOut[key] = tmp2
        return fldOut, area

    nr = fldIn.shape[2] if len(fldIn.shape) > 2 else 1
    nr2 = mask.shape[2] if len(mask.shape) > 2 else 1
    
    if nr2 != nr:
        mask = np.tile(mask, (1, 1, nr))
    
    mask[mask == 0] = np.nan
    # filter for errors
    if len(fldIn.shape)>2:
        tmpshape = fldIn.shape
        #fldIn = fldIn.reshape(tmpshape[0], tmpshape[1])
        fldIn = fldIn.reshape(tmpshape)
        print(fldIn.shape)
    print(mask.shape)
    mask[np.isnan(fldIn)] = np.nan
    areaMask = np.tile(RAC, (1, 1, nr)) * mask
    
    if fldType == "intensive":
        fldOut = np.nansum(fldIn * areaMask) / np.nansum(areaMask)
        area = np.nansum(areaMask)
    else:
        fldOut = np.nansum(fldIn * mask) / np.nansum(areaMask)
        area = np.nansum(areaMask)
    
    return fldOut, area