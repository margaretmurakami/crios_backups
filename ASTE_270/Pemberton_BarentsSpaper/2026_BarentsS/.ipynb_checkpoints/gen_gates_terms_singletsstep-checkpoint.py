## this will be a function attempting to get all of the terms relevant to a single time step in ASTE
# this is taken for example from check_adv_closure which shows that these are similar (we can even add an outprint)

# we need to create a couple of functions
    # create G_T_offline_new
    # create G_S_offline_new

# get_gates3d can be copied here from check_adv_closure

# gateway3d can be copied here from check_adv_closure

############################################################
# define global variables


############################################################
# define functions for getting ADVhT and ADVhS

############################################################
def get_gates3d(ADVx,ADVy,TRACER,nz,ny,nx):

    # define a function to take some ADV term and x and y, and the tracer field THETA and produce only those filled
    # define + as into the basin
    
    # --- reshape to 3D ---
    ADVx = ADVx.reshape((nz, ny, nx))   # advective heat flux on x-faces
    ADVy = ADVy.reshape((nz, ny, nx))   # advective heat flux on y-faces
    TRACER   = TRACER.reshape((nz, ny, nx))     # cell-centered temperature
    
    # ---- BSO ----
    ADV_west = np.zeros((nz, ny, nx))
    # horizontal faces (u-faces)
    for j, i in zip(y_bsoh, x_bsoh):
        # flux through x-face at (j,i) mapped into cell (j,i)
        ADV_west[:, j, i] += ADVx[:, j, i]    # + into basin
    
    # vertical faces (v-faces)
    for j, i in zip(y_bsov, x_bsov):
        # flux through y-face at (j,i) mapped into cell (j-1,i)
        ADV_west[:, j-1, i] -= ADVy[:, j, i]  # sign chosen so + into basin
    
    # ---- FJNZ ----
    ADV_FJNZ = np.zeros((nz, ny, nx))
    ADV_FJNZ[:,y_fjnz,x_fjnzv[0]-1] = -ADVx[:, y_fjnz, x_fjnzv[0]]
    
    # ---- SPFJ (NZ exit) ----
    ADV_SPFJ = np.zeros((nz, ny, nx))
    ADV_SPFJ[:,y_spfjv,x_spfjv-1] -= ADVx[:, y_spfjv, x_spfjv]
    ADV_SPFJ[:,y_spfjh-1,x_spfjh] -= ADVy[:, y_spfjh, x_spfjh]
    ADV_SPFJ[:,y_spfjb-1,x_spfjb] -= ADVy[:, y_spfjb, x_spfjb]
    ADV_SPFJ[:,y_spfjb,x_spfjb-1] -= ADVx[:, y_spfjb, x_spfjb]  # this fixed the issue
    
    # ---- NZRU (small Russia gate) ----
    ADV_NZRU = np.zeros((nz, ny, nx))
    y_nzru_all = np.array([], dtype=int)
    x_nzru_all = np.array([], dtype=int)
    
    for j, i in zip(y_nzruv, x_nzruv):
        ADV_NZRU[:, j, i-1] -= ADVx[:, j, i]   # + into basin
        y_nzru_all = np.append(y_nzru_all, j)
        x_nzru_all = np.append(x_nzru_all, i-1)

    # later we will need to define a small gate for the midway point through the basin and confirm similarity

    return ADV_west, ADV_FJNZ, ADV_SPFJ, ADV_NZRU
    
def _bincount_sum_with_nan(idx, vals, nout):
    """
    NaN-aware per-bin sum:
    if *all* entries in a bin are NaN, that bin returns NaN;
    otherwise NaNs are ignored and finite values are summed.
    """
    # track counts of non-nan contributions
    finite = np.isfinite(vals)
    sums   = np.bincount(idx[finite], vals[finite], minlength=nout).astype(float)
    counts = np.bincount(idx[finite], None, minlength=nout).astype(float)
    out = sums
    out[counts == 0] = np.nan
    return out

# define a function to turn the gates as 3D into 1D for T and S
def gateway3D(ADV_west,ADV_FJNZ,ADV_SPFJ,ADV_NZRU,tracer,binmidTracer,nTm1):
    tracer_flat = tracer.ravel()
    bin_idx_mid = np.digitize(tracer_flat, binmidTracer, right=False) - 1
    valid_mid   = (bin_idx_mid >= 0) & (bin_idx_mid < nTm1) & np.isfinite(tracer_flat)
    idx_mid     = bin_idx_mid[valid_mid]

    # flatten these so we can bin
    ADV_west_flat   = ADV_west.ravel()
    ADV_fjnz_flat   = ADV_FJNZ.ravel()
    ADV_spfj_flat   = ADV_SPFJ.ravel()
    ADV_nzru_flat   = ADV_NZRU.ravel()

    # per-bin sums with NaN-propagation
    ADVh_BSO = _bincount_sum_with_nan(idx_mid, ADV_west_flat[valid_mid], nTm1)
    ADVh_FJNZ = _bincount_sum_with_nan(idx_mid, ADV_fjnz_flat[valid_mid], nTm1)
    ADVh_SPFJ = _bincount_sum_with_nan(idx_mid, ADV_spfj_flat[valid_mid], nTm1)
    ADVh_NZRU = _bincount_sum_with_nan(idx_mid, ADV_nzru_flat[valid_mid], nTm1)
    
    # edge-based G (m^3/s): divide by edge binwidths
    G_BSO = ADVh_BSO #/ binwidthT1
    G_FJNZ = ADVh_FJNZ #/ binwidthT1
    G_SPFJ = ADVh_SPFJ #/ binwidthT1
    G_NZRU = ADVh_NZRU #/ binwidthT1

    return G_BSO,G_FJNZ,G_SPFJ,G_NZRU