def calc_mskmean_T_mod(fldIn, mask, RAC, fldType="intensive"):
    import numpy as np
    
    # If fldIn is a dictionary - run the function again
    if isinstance(fldIn, dict):
        fldOut = {}
        list0 = fldIn.keys()
        for key, value in fldIn.items():
            if isinstance(value, (float, int, np.ndarray)):
                tmp2, area = calc_mskmean_T_mod(value, mask, RAC, fldType)
                fldOut[key] = tmp2
        return fldOut, area

    # if it is not a dictionary, continue
    nr = fldIn.shape[0]
    nr2 = mask.shape[0]
    
    if nr2 != nr:
        mask = np.tile(mask, (nr, 1, 1))

    mask[mask == 0] = np.nan
    mask[fldIn == np.nan] = np.nan
    mask[np.isnan(fldIn)] = np.nan
    areaMask = np.tile(RAC, (nr, 1, 1)) * mask
    
    if fldType == "intensive":
        fldOut = np.nansum(fldIn * areaMask) / np.nansum(areaMask)
        area = np.nansum(areaMask)
    else:
        fldOut = np.nansum(fldIn * mask) / np.nansum(areaMask)
        area = np.nansum(areaMask)
        
    return fldOut, area