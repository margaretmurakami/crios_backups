import numpy as np
import sys
sys.path.append("/home/mmurakami/MITgcm/MITgcm_c68r/MITgcm-checkpoint68r/utils/python/MITgcmutils/MITgcmutils/") # go to parent dir
from mds import *
import os
sys.path.append("/home/mmurakami/crios_backups/an_helper_functions")
sys.path.append("/home/mmurakami/crios_backups/an_helper_functions")
from read_binary import *
from calc_UV_conv_1face import calc_UV_conv_1face
from calc_mskmean_T_mod import calc_mskmean_T_mod
from mk3D_mod import mk3D_mod
from aste_helper_funcs import *
from timing_functions import *           # ts2dte, get_fnames, etc.
from binning import *

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


def _mark_points(mask, xs, ys, code, ny, nx, name="gate"):
    """
    Mark (y, x) points in mask with 'code'.
    If a point already has a different non-NaN code, set it to 3 (overlap).
    Bounds are clipped to the grid silently.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if xs.shape != ys.shape:
        raise ValueError(f"{name}: x/y length mismatch: {xs.shape} vs {ys.shape}")

    # clip to valid indices just in case
    xi = np.clip(xs.astype(int), 0, nx-1)
    yi = np.clip(ys.astype(int), 0, ny-1)

    for j, i in zip(yi, xi):
        cur = mask[j, i]
        if np.isnan(cur):
            mask[j, i] = code
        elif cur == code or cur == 3:
            # already same code or already overlap — leave as is
            continue
        else:
            mask[j, i] = 3  # overlap with different code
    return mask

def create_layers_totalTHETA(tsstr,mygrid,myparms,dirdiags,dirstate,layers_path,mymsk,nz,ny,nx,nfx,nfy):
    ############################################################
    # define the mask here
    # try to use rdmds
    fileprefix = "/scratch3/atnguyen/aste_90x150x60/"
    extBasin='run_template/input_maskTransport/'
    filename = fileprefix + extBasin + "GATE_transports_v2_mskBasin.bin"
    ind = np.fromfile(filename, dtype=np.int32)  # auto-reads .meta for shape/dtype/order
    orig_shape = (ind.shape)
    
    ind2d = ind.reshape(ny,nx)
    
    mymsk = np.full((ny,nx),np.nan)
    mymsk[ind2d == 57408.0] = 1
    
    # make this smaller
    mymsk[:,27:50] = np.nan
    mymsk[:160,12:30] = np.nan
    mymsk[160:163,15:30] = np.nan
    
    ind = ind.reshape(ny,nx)
    mymsk = np.full((ny,nx),np.nan)
    mymsk[ind == 57408.0] = 1
    
    # make this smaller
    mymsk[:,27:50] = np.nan
    mymsk[:160,12:30] = np.nan
    mymsk[160:163,15:30] = np.nan

    ysmsk,xsmsk = np.where(mymsk==1)[0],np.where(mymsk==1)[1]

    # define the gates for the miniaste

    # these are the indices we want to read from, but not write to
    # at y = 186, we want -ADVy
    x_bsoh = np.array([54, 54, 54, 54, 54])
    x_bsov = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9])    # from Norway to Sp
    y_bsoh = np.array([145, 146, 147, 148, 149])
    y_bsov = np.array([186, 186, 186, 186, 186, 186, 186, 186, 186, 186])    # from Norway to Sp
    
    x_spfjh = np.array([20,20,20,23,24,24,26,26])    # vertical gates from Sp to Fj as -x
    y_spfjh = np.array([185,184,183,182,181,180,179,178])
    x_spfjv = np.array([20,21,22,23,24,25,26])          # horizontal gates where we want to read -y
    y_spfjv = np.array([183,183,183,183,182,180,180])
    
    y_fjnzv = np.arange(165,175,1)
    x_fjnzv = np.full_like(y_fjnzv,27)    # horizontal gate where we want to read -x
    
    y_nzruv = np.arange(152,155,1)
    x_nzruv = np.full_like(y_nzruv,12)   # horizontal gate where we want to read -x

    # gates_mask starts as NaN everywhere
    gates_mask = np.full((ny, nx), np.nan, dtype=float)
    
    # ---- mark H gates with code = 1 ----
    gates_mask = _mark_points(gates_mask, x_bsoh, y_bsoh, 1, ny, nx, name="bsoh")
    gates_mask = _mark_points(gates_mask, x_spfjh, y_spfjh, 1, ny, nx, name="spfjh")
    
    # ---- mark V gates with code = 2 ----
    gates_mask = _mark_points(gates_mask, x_bsov,  y_bsov,  2, ny, nx, name="bsov")
    gates_mask = _mark_points(gates_mask, x_spfjv, y_spfjv, 2, ny, nx, name="spfjv")
    gates_mask = _mark_points(gates_mask, x_fjnzv, y_fjnzv, 1, ny, nx, name="fjnzv")
    gates_mask = _mark_points(gates_mask, x_nzruv, y_nzruv, 1, ny, nx, name="nzruv")
    
    # Optional: if you prefer 0 instead of NaN for “not a gate”
    # gates_mask = np.nan_to_num(gates_mask, nan=0.0)

    # let's make a mask of these to double check that we did this correctly
    gates_mask[182,23] = 3
    gates_mask[180,24] = 3 
    gates_mask[180,26] = np.nan
    gates_mask[182,24] = np.nan
    gates_mask[183,23] = np.nan

    
    testmsk = gates_mask.copy()
    testmsk[:,:19] = np.nan
    testmsk[:,30:] = np.nan
    testmsk[:178,:] = np.nan
    y_spfjv2,x_spfjv2 = np.where(testmsk == 2)[0],np.where(testmsk == 2)[1]
    y_spfjh2,x_spfjh2 = np.where(testmsk == 1)[0],np.where(testmsk == 1)[1]
    y_spfjb2,x_spfjb2 = np.where(testmsk == 3)[0],np.where(testmsk == 3)[1]
    RAC = mygrid['RAC']

    ############################################################
    # from tsstr, loop through and generate the actual values from the output

    # define the layers
    filename = "layers2TH"
    boundsT = rdmds(layers_path + filename)
    binsTH_edges = boundsT.reshape(boundsT.shape[0])
    binsTH_centers = (binsTH_edges[:-1] + binsTH_edges[1:])/2
    nT = binsTH_edges.shape[0]-1
    
    filename = "layers1SLT"
    boundsS = rdmds(layers_path + filename)
    binsSLT_edges = boundsS.reshape(boundsS.shape[0])
    binsSLT_centers = (binsSLT_edges[:-1] + binsSLT_edges[1:])/2
    nS = binsSLT_edges.shape[0]-1
    
    binwidthT = binsTH_edges[1:] - binsTH_edges[:-1]
    binwidthS = binsSLT_edges[1:] - binsSLT_edges[:-1]
    
    binwidthT1 = (binwidthT[:-1] + binwidthT[1:])/2
    binwidthS1 = (binwidthS[:-1] + binwidthS[1:])/2
    
    binmidT = ((boundsT[:-1] + boundsT[1:])/2).reshape(nT)
    binmidS = ((boundsS[:-1] + boundsS[1:])/2).reshape(nT)

    # read from T and S
    t2 = int(tsstr[1])
    mymsk3d = np.tile(mymsk[np.newaxis,:,:],(nz,1,1))

    # 'diags/state_3d_set1'
    # read theta and salt averages from the t2 timestep (average)
    file_name = "state_3d_set1"
    meta_state_3d_set1 = parsemeta(dirstate + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_state_3d_set1["fldList"])
    varnames = np.array(["THETA","SALT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    THETA,its,meta = rdmds(os.path.join(dirstate, file_name),t2,returnmeta=True,rec=recs[0])
    SALT,its,meta = rdmds(os.path.join(dirstate, file_name),t2,returnmeta=True,rec=recs[1])
    
    THETA = THETA.reshape(nz,ny,nx)
    SALT = SALT.reshape(nz,ny,nx)


    file_name = "budg3d_hflux_set2"
    meta_budg3d_hflux_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_hflux_set2["fldList"])
    varnames = np.array(["ADVx_TH","ADVy_TH"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    ADVx_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    ADVy_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    
    # now 3d zfluxes
    file_name = "budg3d_zflux_set1"
    meta_budg3d_zflux_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_zflux_set1["fldList"])
    varnames = np.array(["ADVr_TH"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    ADVr_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    ADVr_TH = ADVr_TH.reshape(nz,ny,nx)
    
    hf = mygrid['hFacC']
    
    ADV_hconv = calc_UV_conv_mod(nfx, nfy,get_aste_faces(ADVx_TH.reshape(nz, ny, nx), nfx, nfy),get_aste_faces(ADVy_TH.reshape(nz, ny, nx), nfx, nfy))
    ADV_hconv = ADV_hconv * hf   # degC·m^3/s at cell centers (matches: ff.DFh = ff.DFh .* hf)
    ADVhT = ADV_hconv
    
    trWtopADV = -(ADVr_TH)
    
    ADVrT = np.zeros((nz,ny,nx),dtype=float)
    ADVrT[:-1,:,:] = (trWtopADV[:-1] - trWtopADV[1:])  # this is not the way we did it in the original code but this is the way An has done it so we try

    # do this manually from ADVh
    nT   = boundsT.size - 1
    nTm1 = nT - 1
    nS = boundsS.size -1 
    nSm1 = nS - 1
    
    # mask by the Barents Sea
    ADVhT_BS = ADVhT * mymsk3d
    ADVrT_BS = ADVrT * mymsk3d
    
    
    # --- "new" interpretation: bin by binmidT intervals ---
    theta_flat = THETA.ravel()
    salt_flat = SALT.ravel()
    ADVh_flat   = ADVhT_BS.ravel()
    ADVr_flat   = ADVrT_BS.ravel()
    
    # binmidT[i] <= THETA < binmidT[i+1], i=0..nT-2
    bin_idx_mid = np.digitize(theta_flat, binmidT, right=False) - 1
    valid_mid   = (bin_idx_mid >= 0) & (bin_idx_mid < nTm1) & np.isfinite(theta_flat)
    idx_mid     = bin_idx_mid[valid_mid]
    bin_idx_midS = np.digitize(salt_flat, binmidS, right=False) - 1
    valid_midS  = (bin_idx_midS >= 0) & (bin_idx_midS < nSm1) & np.isfinite(salt_flat)
    idx_midS     = bin_idx_midS[valid_midS]
    
    # per-bin sums with NaN-propagation
    ADVh_new = _bincount_sum_with_nan(idx_mid, ADVh_flat[valid_mid], nTm1)
    ADVr_new = _bincount_sum_with_nan(idx_mid, ADVr_flat[valid_mid], nTm1)
    
    
    # edge-based G (m^3/s): divide by edge binwidths
    G_off_new_h = ADVh_new / binwidthT1
    G_off_new_r = ADVr_new / binwidthT1

    Tbin,Sbin = np.meshgrid(binsTH_centers,binsSLT_centers)

    # we want to bin theta and salt into the T and S bins
    binned_theta = bin_array(THETA,binsTH_centers)
    binned_theta = binned_theta.astype(float)
    binned_theta[binned_theta == nT] = np.nan     # because the binning is setting nan to last value
    binned_salinity = bin_array(SALT,binsSLT_centers)
    binned_salinity = binned_salinity.astype(float)
    binned_salinity[binned_salinity == nS] = np.nan

    y_bso_all = np.array([]).astype(int)
    x_bso_all = np.array([]).astype(int)

    
    # --- reshape to 3D ---
    ADVx_TH = ADVx_TH.reshape((nz, ny, nx))   # advective heat flux on x-faces
    ADVy_TH = ADVy_TH.reshape((nz, ny, nx))   # advective heat flux on y-faces
    THETA   = THETA.reshape((nz, ny, nx))     # cell-centered temperature
    
    # ------------------------------------------------------------------
    # 1. Build theta at faces
    # ------------------------------------------------------------------
    
    # x-faces: between (i-1, i) along x
    theta_x = np.zeros_like(ADVx_TH)
    theta_x[:, :, 1:] = 0.5 * (THETA[:, :, 1:] + THETA[:, :, :-1])
    theta_x[:, :, 0]  = theta_x[:, :, 1]      # simple fill for western boundary
    
    # y-faces: between (j-1, j) along y
    theta_y = np.zeros_like(ADVy_TH)
    theta_y[:, 1:, :] = 0.5 * (THETA[:, 1:, :] + THETA[:, :-1, :])
    theta_y[:, 0, :]  = theta_y[:, 1, :]      # simple fill for southern boundary
    
    # ------------------------------------------------------------------
    # 2. Convert heat flux (degC·m^3/s) -> volume flux (m^3/s)
    #    q_vol = q_heat / theta_face
    # ------------------------------------------------------------------
    
    eps = 1e-6  # to avoid divide-by-zero in very cold cells
    
    ADVx_vol = np.zeros_like(ADVx_TH)
    mask_x   = np.isfinite(theta_x) & (np.abs(theta_x) > eps)
    ADVx_vol[mask_x] = ADVx_TH[mask_x] #/ theta_x[mask_x]
    
    ADVy_vol = np.zeros_like(ADVy_TH)
    mask_y   = np.isfinite(theta_y) & (np.abs(theta_y) > eps)
    ADVy_vol[mask_y] = ADVy_TH[mask_y] #/ theta_y[mask_y]
    
    # bolus
    ADVx_vol = np.zeros_like(ADVx_TH)
    mask_x   = np.isfinite(theta_x) & (np.abs(theta_x) > eps)
    ADVx_vol[mask_x] = (ADVx_TH[mask_x]) #/ theta_x[mask_x]
    
    ADVy_vol = np.zeros_like(ADVy_TH)
    mask_y   = np.isfinite(theta_y) & (np.abs(theta_y) > eps)
    ADVy_vol[mask_y] = (ADVy_TH[mask_y]) #/ theta_y[mask_y]
    
    # ------------------------------------------------------------------
    # 3. Build gateway transports using volume fluxes at the faces
    #    Sign convention: comments assume "positive into basin"
    # ------------------------------------------------------------------
    
    # ADVx_vol = ADVx_TH
    # ADVy_vol = ADVy_TH
    
    # ---- BSO ----
    ADV_west = np.zeros((nz, ny, nx))
    y_bso_all = np.array([], dtype=int)
    x_bso_all = np.array([], dtype=int)
    
    # horizontal faces (u-faces)
    for j, i in zip(y_bsoh, x_bsoh):
        # flux through x-face at (j,i) mapped into cell (j,i)
        ADV_west[:, j, i] += ADVx_vol[:, j, i]    # + into basin
        y_bso_all = np.append(y_bso_all, j)
        x_bso_all = np.append(x_bso_all, i)
    
    # vertical faces (v-faces)
    for j, i in zip(y_bsov, x_bsov):
        # flux through y-face at (j,i) mapped into cell (j-1,i)
        ADV_west[:, j-1, i] -= ADVy_vol[:, j, i]  # sign chosen so + into basin
        y_bso_all = np.append(y_bso_all, j-1)
        x_bso_all = np.append(x_bso_all, i)
    
    # ---- FJNZ ----
    ADV_FJNZ = np.zeros((nz, ny, nx))
    y_fjnz_all = np.array([], dtype=int)
    x_fjnz_all = np.array([], dtype=int)
    
    for j, i in zip(y_fjnzv, x_fjnzv):
        # x-face at (j,i) mapped into (j, i-1), + into basin
        ADV_FJNZ[:, j, i-1] -= ADVx_vol[:, j, i]
        y_fjnz_all = np.append(y_fjnz_all, j)
        x_fjnz_all = np.append(x_fjnz_all, i-1)
    
    # ---- SPFJ (NZ exit) ----
    ADV_SPFJ = np.zeros((nz, ny, nx))
    y_spfj_all = np.array([], dtype=int)
    x_spfj_all = np.array([], dtype=int)
    
    # # CHANGED
    for j,i in zip(y_spfjv2,x_spfjv2):
        ADV_SPFJ[:, j-1, i] -= ADVy_vol[:, j, i]
        y_spfj_all = np.append(y_spfj_all, j-1)
        x_spfj_all = np.append(x_spfj_all, i)
    
    for j,i in zip(y_spfjh2,x_spfjh2):
        ADV_SPFJ[:, j, i-1] -= ADVx_vol[:, j, i]
        y_spfj_all = np.append(y_spfj_all, j)
        x_spfj_all = np.append(x_spfj_all, i-1)
        
    for j,i in zip(y_spfjb2,x_spfjb2):
        ADV_SPFJ[:, j, i-1] -= ADVx_vol[:, j, i]
        y_spfj_all = np.append(y_spfj_all, j)
        x_spfj_all = np.append(x_spfj_all, i-1)
        ADV_SPFJ[:, j-1, i] -= ADVy_vol[:, j, i]
        y_spfj_all = np.append(y_spfj_all, j-1)
        x_spfj_all = np.append(x_spfj_all, i)
    
    # ---- NZRU (small Russia gate) ----
    ADV_NZRU = np.zeros((nz, ny, nx))
    y_nzru_all = np.array([], dtype=int)
    x_nzru_all = np.array([], dtype=int)
    
    for j, i in zip(y_nzruv, x_nzruv):
        ADV_NZRU[:, j, i-1] -= ADVx_vol[:, j, i]   # + into basin
        y_nzru_all = np.append(y_nzru_all, j)
        x_nzru_all = np.append(x_nzru_all, i-1)

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
    # this is not correct because we want to divide by Face T
    G_BSO = ADVh_BSO / binwidthT1
    G_FJNZ = ADVh_FJNZ / binwidthT1
    G_SPFJ = ADVh_SPFJ / binwidthT1
    G_NZRU = ADVh_NZRU / binwidthT1

    Msum = (ADVh_BSO + ADVh_FJNZ + ADVh_SPFJ + ADVh_NZRU)

    # calculate the other terms in the eqn for budget

    # load the other terms from the offline version -- we can just put this on top of the T diagram
    
    ## do the advective convergence
    file_name = "budg3d_hflux_set2"
    meta_budg3d_hflux_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_hflux_set2["fldList"])
    varnames = np.array(["DFxE_TH","DFyE_TH"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    DFxE_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    DFyE_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    
    
    # now 3d zfluxes
    file_name = "budg3d_zflux_set1"
    meta_budg3d_zflux_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_zflux_set1["fldList"])
    varnames = np.array(["DFrE_TH","DFrI_TH"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    DFrE_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    DFrI_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    DFrE_TH = DFrE_TH.reshape(nz,ny,nx)
    DFrI_TH = DFrI_TH.reshape(nz,ny,nx)
    
    DF_hconv = calc_UV_conv_mod(nfx, nfy,get_aste_faces(DFxE_TH.reshape(nz, ny, nx), nfx, nfy),get_aste_faces(DFyE_TH.reshape(nz, ny, nx), nfx, nfy))
    DF_hconv = DF_hconv * hf   # degC·m^3/s at cell centers (matches: ff.DFh = ff.DFh .* hf)
    DFhT = DF_hconv
    
    trWtopDF = -(DFrE_TH+DFrI_TH)
    
    DFrT = np.zeros((nz,ny,nx),dtype=float)
    DFrT[:-1,:,:] = (trWtopDF[:-1] - trWtopDF[1:])
    
    # to get the surface term, we need J/s and convert to degC.m^3/s
    file_name = 'budg2d_zflux_set1'
    meta_budg2d_zflux_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg2d_zflux_set1["fldList"])
    varnames = np.array(["TFLUX","oceQsw","SItflux"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    TFLUX,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    oceQsw,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    SItflux,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[2])
    TFLUX = TFLUX.reshape(ny,nx)
    oceQsw = oceQsw.reshape(ny,nx)
    SItflux = SItflux.reshape(ny,nx)
    
    # we need to create zconv_top and swtop
    dd = mygrid['RF'][:-1]
    swfrac = 0.62*np.exp(dd/0.6)+(1-0.62)*np.exp(dd/20)
    swfrac[dd < -200] = 0
    swtop=mk3D_mod(swfrac,np.zeros((nz,ny,nx)))*mk3D_mod(RAC*oceQsw,np.zeros((nz,ny,nx)))   # J/s
    
    # zconvtop_heat is here
    zconv_top_heat = TFLUX * RAC     # W/m^2 * m^2 = J/s
    
    
    def surface_contrib_JT(zconv_top_heat, swtop, rcp, fill_last=0.0):
        """
        zconv_top_heat: (ny, nx)
        swtop:          (nz, ny, nx)
        rcp:            scalar
        fill_last:      value for bottom slice (k = nz-1), usually 0.0 or np.nan
        returns:
          JsurfT:       (nz, ny, nx)  # Sv / PSU
        """
        nz, ny, nx = swtop.shape
    
        eT = zconv_top_heat.reshape(1, ny, nx)  # (1,ny,nx) for broadcast
    
        J = np.empty_like(swtop, dtype=float)
    
        # k = 0: (eT - fT[1]) / rcp / dT / dS * 1e-6
        J[0] = (eT[0] - swtop[1]) / rcp if np.ndim(binwidthT)==0 else \
               (eT[0] - swtop[1]) / rcp
    
        # 1 .. nz-2: -(fT[k+1]-fT[k]) / rcp / dT / dS * 1e-6
        J[1:nz-1] = -(swtop[2:nz] - swtop[1:nz-1]) / rcp
    
        # bottom slice (k = nz-1): no k+1; choose your boundary convention
        J[-1] = fill_last
        return J
    
    Ft_surftest = surface_contrib_JT(zconv_top_heat,swtop,myparms['rcp'])    # this is in degC.m^3/s
    
    # read kpp tend and from 3d zflux
    file_name = "budg3d_kpptend_set1"
    meta_budg3d_kpptend_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_kpptend_set1["fldList"])
    varnames = np.array(["KPPg_TH"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    KPPg_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    KPPg_TH = KPPg_TH.reshape(nz,ny,nx)
    
    # do the vertical convergence for KPP
    trWtopKPP = -(KPPg_TH)         # degC.m^3/s
    
    tmpkpp = np.full((nz,ny,nx),np.nan)
    tmpkpp[:-1,:,:] = trWtopKPP[:-1] - trWtopKPP[1:]
    
    
    # load the tend from the get_Jterms and plot this
    file_name = 'budg3d_snap_set2'
    meta_budg3d_snap_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_snap_set2["fldList"])
    varnames = np.array(["THETADR"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    
    THETADR = np.full((len(tsstr),nz,ny,nx),np.nan)
    for i in range(len(tsstr)):
        thisTHETADR,its,meta = rdmds(os.path.join(dirdiags, file_name),int(tsstr[i]),returnmeta=True,rec=recs[0])
        thisTHETADR = thisTHETADR.reshape(nz,ny,nx)
        THETADR[i] = thisTHETADR
    
    THETADR =  (THETADR[1, :, :,:] - THETADR[0, :,:, :]) / 1    # degC.m/
    AB_gT = 0
    tmptend=(THETADR-AB_gT)*mk3D_mod(RAC,THETADR)   # degC.m/s * m^2 = degC.m^3/s
    tmptend = tmptend                          # degC.m^3/s

    # redefine all the terms as a list from how we did before

    # define the ADVh total for this mymsk2
    G_T_offline_new = np.zeros((7, nT-1))
    dF_Tnew = np.zeros((7, nT-1))
    Lijnew = np.zeros((7, nT-1), dtype=int)
    
    # also mask these by mymsk3
    # flatten the 3D arrays along all dimensions, as MATLAB’s tmp(:) does
    T_flat    = np.ravel(THETA* mymsk3d, order='F')
    ADVh_flat = np.ravel(ADVhT* mymsk3d,  order='F')
    ADVr_flat = np.ravel(ADVrT* mymsk3d,  order='F')
    DFh_flat = np.ravel(DFhT* mymsk3d,  order='F')
    DFr_flat = np.ravel(DFrT* mymsk3d,  order='F')
    surf_flat = np.ravel(Ft_surftest* mymsk3d,  order='F')
    kpp_flat = np.ravel(tmpkpp* mymsk3d,  order='F')
    tend_flat = np.ravel(tmptend* mymsk3d,  order='F')
    
    for i in range(nT-1):
        # MATLAB: ij = find(tmp(:) >= bbb.binmidT(i) & tmp(:) < bbb.binmidT(i+1))
        ij = np.where((T_flat >= binmidT[i]) & (T_flat < binmidT[i + 1]))[0]
        Lijnew[0, i] = len(ij)
    
        if len(ij) > 0:
            # MATLAB: dF_Tnew(4,i)=sum(ff.advh(ij)); dF_Tnew(5,i)=sum(ff.advr(ij));
            dF_Tnew[0, i] = np.nansum(ADVh_flat[ij])
            dF_Tnew[1, i] = np.nansum(ADVr_flat[ij])
            dF_Tnew[2, i] = np.nansum(DFh_flat[ij])
            dF_Tnew[3, i] = np.nansum(DFr_flat[ij])
            dF_Tnew[4, i] = np.nansum(surf_flat[ij])
            dF_Tnew[5, i] = np.nansum(kpp_flat[ij])
            dF_Tnew[6, i] = np.nansum(tend_flat[ij])
    
    # MATLAB: G_T_offline_new = dF_Tnew ./ repmat(bbb.binwidthT1,[6 1])
    G_T_offline_new = dF_Tnew / binwidthT1[None, :]

    return Msum, dF_Tnew

# def create_layers_totalSALT(tsstr,mygrid,myparms,dirdiags,dirstate,layers_path,mymsk,nz,ny,nx,nfx,nfy):
