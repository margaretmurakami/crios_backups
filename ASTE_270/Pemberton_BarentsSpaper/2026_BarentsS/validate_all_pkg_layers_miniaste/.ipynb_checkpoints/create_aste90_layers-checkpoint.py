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

def bincount_sum_zero(idx, val, nbins):
    """
    Like your _bincount_sum_with_nan, but returns zeros and ignores NaNs in val.
    """
    out = np.zeros((nbins,), dtype=float)
    idx = np.asarray(idx)
    val = np.asarray(val, dtype=float)

    good = (idx >= 0) & (idx < nbins) & np.isfinite(val)
    if np.any(good):
        np.add.at(out, idx[good], val[good])
    return out


def bin_gate_by_face_theta_zero(theta_face_1d, flux_1d, bin_edges, nbins):
    theta_face_1d = np.asarray(theta_face_1d)
    flux_1d       = np.asarray(flux_1d, dtype=float)

    b = np.digitize(theta_face_1d, bin_edges, right=False) - 1  # 0..nbins-1
    valid = (b >= 0) & (b < nbins) & np.isfinite(theta_face_1d) & np.isfinite(flux_1d)

    return bincount_sum_zero(b[valid], flux_1d[valid], nbins)


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

# add a function to bin to TS space for the gateways
def bin_gate_by_face_TS_zero(theta_face, salt_face, flux_face,
                             binmidT, binmidS, nTm1, nSm1):
    """
    Bin face flux samples into T-S space.

    Parameters
    ----------
    theta_face : 1D array
        Face-averaged temperature samples.
    salt_face : 1D array
        Face-averaged salinity samples.
    flux_face : 1D array
        Face flux samples corresponding to (theta_face, salt_face).
    binmidT : 1D array
        Temperature bin centers or edges used consistently with searchsorted logic.
    binmidS : 1D array
        Salinity bin centers or edges used consistently with searchsorted logic.
    nTm1 : int
        Number of T bins in the output.
    nSm1 : int
        Number of S bins in the output.

    Returns
    -------
    out : 2D array, shape (nTm1, nSm1)
        Zero-filled T-S binned sum of flux_face.
    """
    out = np.zeros((nTm1, nSm1), dtype=float)

    valid = (
        np.isfinite(theta_face) &
        np.isfinite(salt_face) &
        np.isfinite(flux_face)
    )

    if not np.any(valid):
        return out

    theta_face = theta_face[valid]
    salt_face  = salt_face[valid]
    flux_face  = flux_face[valid]

    iT = np.searchsorted(binmidT, theta_face, side="right") - 1
    iS = np.searchsorted(binmidS, salt_face,  side="right") - 1

    ok = (iT >= 0) & (iT < nTm1) & (iS >= 0) & (iS < nSm1)

    if np.any(ok):
        np.add.at(out, (iT[ok], iS[ok]), flux_face[ok])

    return out

def create_layers_totalTHETA(tsstr,mygrid,myparms,dirdiags,dirstate,layers_path,mymsk,nz,ny,nx,nfx,nfy,dt,boundsT,boundsS, mapping=False,debug=False):
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

    # define the layers -- change this instead to read from function
    binsTH_edges = boundsT.reshape(boundsT.shape[0])
    binsTH_centers = (binsTH_edges[:-1] + binsTH_edges[1:])/2
    nT = binsTH_edges.shape[0]-1
    
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

    if debug:
        mymsk = np.full((ny,nx),np.nan)
        mymsk[150,19] = 1
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

    ############################################################################################################
    # --- reshape to 3D ---
    ADVx_TH = ADVx_TH.reshape((nz, ny, nx))   # advective heat flux on x-faces
    ADVy_TH = ADVy_TH.reshape((nz, ny, nx))   # advective heat flux on y-faces
    THETA   = THETA.reshape((nz, ny, nx))     # cell-centered temperature
    SALT = SALT.reshape((nz, ny, nx))
    
    # ------------------------------------------------------------------
    # 1. Build theta at faces
    # ------------------------------------------------------------------
    
    # x-faces: between (i-1, i) along x
    theta_x = np.zeros_like(ADVx_TH)
    theta_x[:, :, 1:] = 0.5 * (THETA[:, :, 1:] + THETA[:, :, :-1])
    theta_x[:, :, 0]  = theta_x[:, :, 1]      # simple fill for western boundary 
    salt_x = np.zeros_like(ADVx_TH)
    salt_x[:, :, 1:] = 0.5 * (SALT[:, :, 1:] + SALT[:, :, :-1])
    salt_x[:, :, 0]  = salt_x[:, :, 1]
    
    # y-faces: between (j-1, j) along y
    theta_y = np.zeros_like(ADVy_TH)
    theta_y[:, 1:, :] = 0.5 * (THETA[:, 1:, :] + THETA[:, :-1, :])
    theta_y[:, 0, :]  = theta_y[:, 1, :]      # simple fill for southern boundary
    salt_y = np.zeros_like(ADVy_TH)
    salt_y[:, 1:, :] = 0.5 * (SALT[:, 1:, :] + SALT[:, :-1, :])
    salt_y[:, 0, :]  = salt_y[:, 1, :]
    
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
        
        
    # ------------------------------------------------------------
    # Build per-gate (theta_face, flux) samples and bin -> arrays are ZERO-filled
    # ------------------------------------------------------------
    
    # ---- BSO ----
    theta_BSO_list, salt_BSO_list ,flux_BSO_list = [], [], []
    
    for j, i in zip(y_bsoh, x_bsoh):
        theta_BSO_list.append(theta_x[:, j, i].ravel())
        salt_BSO_list.append(salt_x[:, j, i].ravel())
        flux_BSO_list.append( ADVx_vol[:, j, i].ravel() )
    
    for j, i in zip(y_bsov, x_bsov):
        theta_BSO_list.append(theta_y[:, j, i].ravel())
        salt_BSO_list.append(salt_y[:,j,i].ravel())
        flux_BSO_list.append( (-ADVy_vol[:, j, i]).ravel() )
    
    theta_BSO = np.concatenate(theta_BSO_list) if theta_BSO_list else np.array([], dtype=float)
    salt_BSO = np.concatenate(salt_BSO_list) if salt_BSO_list else np.array([], dtype=float)
    flux_BSO  = np.concatenate(flux_BSO_list)  if flux_BSO_list  else np.array([], dtype=float)

    # add a section for the M term, or plotted in T space
    ADVh_BSO = bin_gate_by_face_theta_zero(theta_BSO, flux_BSO, binmidT, nTm1)
    G_BSO    = ADVh_BSO / binwidthT1      # in units of m^3/s
    
    ADVh_BSO_TS = bin_gate_by_face_TS_zero(theta_BSO, salt_BSO, flux_BSO,binmidT, binmidS, nTm1, nSm1)
    G_BSO_TS = ADVh_BSO_TS / (binwidthT1[:, None] * binwidthS1[None, :])  # in units of m^3/s/PSU
    
    
    # ---- FJNZ ----
    theta_FJNZ_list, salt_FJNZ_list, flux_FJNZ_list = [], [], []
    
    for j, i in zip(y_fjnzv, x_fjnzv):
        theta_FJNZ_list.append(theta_x[:, j, i].ravel())
        salt_FJNZ_list.append(salt_x[:, j, i].ravel())
        flux_FJNZ_list.append((-ADVx_vol[:, j, i]).ravel())
    
    theta_FJNZ = np.concatenate(theta_FJNZ_list) if theta_FJNZ_list else np.array([], dtype=float)
    salt_FJNZ  = np.concatenate(salt_FJNZ_list)  if salt_FJNZ_list  else np.array([], dtype=float)
    flux_FJNZ  = np.concatenate(flux_FJNZ_list)  if flux_FJNZ_list  else np.array([], dtype=float)
    
    ADVh_FJNZ = bin_gate_by_face_theta_zero(theta_FJNZ, flux_FJNZ, binmidT, nTm1)
    G_FJNZ    = ADVh_FJNZ / binwidthT1

    ADVh_FJNZ_TS = bin_gate_by_face_TS_zero(theta_FJNZ, salt_FJNZ, flux_FJNZ,binmidT, binmidS, nTm1, nSm1)
    G_FJNZ_TS = ADVh_FJNZ_TS / (binwidthT1[:, None] * binwidthS1[None, :])  # m^3/s/PSU
    
    
    # ---- SPFJ ----
    theta_SPFJ_list, salt_SPFJ_list, flux_SPFJ_list = [], [], []
    
    for j, i in zip(y_spfjv2, x_spfjv2):
        theta_SPFJ_list.append(theta_y[:, j, i].ravel())
        salt_SPFJ_list.append(salt_y[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVy_vol[:, j, i]).ravel())
    
    for j, i in zip(y_spfjh2, x_spfjh2):
        theta_SPFJ_list.append(theta_x[:, j, i].ravel())
        salt_SPFJ_list.append(salt_x[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVx_vol[:, j, i]).ravel())
    
    for j, i in zip(y_spfjb2, x_spfjb2):
        theta_SPFJ_list.append(theta_x[:, j, i].ravel())
        salt_SPFJ_list.append(salt_x[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVx_vol[:, j, i]).ravel())
    
        theta_SPFJ_list.append(theta_y[:, j, i].ravel())
        salt_SPFJ_list.append(salt_y[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVy_vol[:, j, i]).ravel())
    
    theta_SPFJ = np.concatenate(theta_SPFJ_list) if theta_SPFJ_list else np.array([], dtype=float)
    salt_SPFJ  = np.concatenate(salt_SPFJ_list)  if salt_SPFJ_list  else np.array([], dtype=float)
    flux_SPFJ  = np.concatenate(flux_SPFJ_list)  if flux_SPFJ_list  else np.array([], dtype=float)
    
    ADVh_SPFJ = bin_gate_by_face_theta_zero(theta_SPFJ, flux_SPFJ, binmidT, nTm1)
    G_SPFJ    = ADVh_SPFJ / binwidthT1

    ADVh_SPFJ_TS = bin_gate_by_face_TS_zero(theta_SPFJ, salt_SPFJ, flux_SPFJ,binmidT, binmidS, nTm1, nSm1)
    G_SPFJ_TS = ADVh_SPFJ_TS / (binwidthT1[:, None] * binwidthS1[None, :])
    
    # ---- NZRU ----
    theta_NZRU_list, salt_NZRU_list, flux_NZRU_list = [], [], []
    
    for j, i in zip(y_nzruv, x_nzruv):
        theta_NZRU_list.append(theta_x[:, j, i].ravel())
        salt_NZRU_list.append(salt_x[:, j, i].ravel())
        flux_NZRU_list.append((-ADVx_vol[:, j, i]).ravel())
    
    theta_NZRU = np.concatenate(theta_NZRU_list) if theta_NZRU_list else np.array([], dtype=float)
    salt_NZRU  = np.concatenate(salt_NZRU_list)  if salt_NZRU_list  else np.array([], dtype=float)
    flux_NZRU  = np.concatenate(flux_NZRU_list)  if flux_NZRU_list  else np.array([], dtype=float)
    
    ADVh_NZRU = bin_gate_by_face_theta_zero(theta_NZRU, flux_NZRU, binmidT, nTm1)
    G_NZRU    = ADVh_NZRU / binwidthT1
    ADVh_NZRU_TS = bin_gate_by_face_TS_zero(theta_NZRU, salt_NZRU, flux_NZRU,binmidT, binmidS, nTm1, nSm1) 
    G_NZRU_TS = ADVh_NZRU_TS / (binwidthT1[:, None] * binwidthS1[None, :])
    

    # instead of returning Msum here, make Msum a dictionary with these four gates
    Msum = {}
    Msum["BSO"]  = ADVh_BSO
    Msum["FJNZ"] = ADVh_FJNZ
    Msum["SPFJ"] = ADVh_SPFJ
    Msum["NZRU"] = ADVh_NZRU
    Msum["Msum"] = ADVh_BSO + ADVh_FJNZ + ADVh_SPFJ + ADVh_NZRU  # degC.m^3/s
    
    # add the not-normalized versions of this
    Msum["BSO_TSnn"] = ADVh_BSO_TS
    Msum["FJNZ_TSnn"] = ADVh_FJNZ_TS
    Msum["SPFJ_TSnn"] = ADVh_SPFJ_TS
    Msum["NZRU_TSnn"] = ADVh_NZRU_TS  # we can run a validation with these first and then see
    Msum["Msum_TSnn"] = ADVh_BSO_TS + ADVh_FJNZ_TS + ADVh_SPFJ_TS + ADVh_NZRU_TS  # degC.m^3/s

    # add the normalized versions of this to get the vector
    Msum["BSO_TS"] = G_BSO_TS
    Msum["FJNZ_TS"] = G_FJNZ_TS
    Msum["SPFJ_TS"] = G_SPFJ_TS
    Msum["NZRU_TS"] = G_NZRU_TS
    Msum["Msum_TS"] = G_BSO_TS + G_FJNZ_TS + G_SPFJ_TS + G_NZRU_TS  # m^3/s/PSU
    
    ############################################################################################################
    # modify the above to bin by face THETA rather than cell-center THETA

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
    # we need to create zconv_top and swtop
    dd = mygrid['RF'][:-1]
    swfrac = 0.62*np.exp(dd/0.6) + (1-0.62)*np.exp(dd/20)
    swfrac[dd < -200] = 0
    
    # shortwave penetration profile in J/s
    swtop = (
        mk3D_mod(swfrac, np.zeros((nz, ny, nx))) *
        mk3D_mod(RAC * oceQsw, np.zeros((nz, ny, nx)))
    )
    
    # non-penetrative surface heat input in J/s
    zconv_top_heat_total = TFLUX * RAC
    
    
    def surface_contrib_JT(zconv_top_heat, swtop, rcp, fill_last=0.0):
        """
        zconv_top_heat: (ny, nx)     surface convergence term in J/s
        swtop:          (nz, ny, nx) penetrating downward flux profile in J/s
        returns:
          JsurfT:       (nz, ny, nx) in degC.m^3/s
        """
        nz_, ny_, nx_ = swtop.shape
        eT = zconv_top_heat.reshape(1, ny_, nx_)
        J = np.empty_like(swtop, dtype=float)
    
        # top cell: net surface input minus flux penetrating into cell below
        J[0] = (eT[0] - swtop[1]) / rcp
    
        # interior cells: convergence of penetrating shortwave flux
        J[1:nz_-1] = -(swtop[2:nz_] - swtop[1:nz_-1]) / rcp
    
        # bottom boundary convention
        J[-1] = fill_last
    
        return J
    
    
    ############################################################
    # original total surface contribution
    Ft_surf = surface_contrib_JT(
        zconv_top_heat_total,
        swtop,
        myparms['rcp']
    )
    
    # split into just two terms:
    # 1) zconv_top contribution alone
    Ft_zconv_top = surface_contrib_JT(
        zconv_top_heat_total,
        np.zeros((nz, ny, nx)),
        myparms['rcp']
    )
    
    # 2) shortwave penetration contribution alone
    Ft_swtop = surface_contrib_JT(
        np.zeros((ny, nx)),
        swtop,
        myparms['rcp']
    )    # this is in degC.m^3/s
    
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
    
    THETADR =  (THETADR[1, :, :,:] - THETADR[0, :,:, :]) / dt    # degC.m/
    AB_gT = 0
    tmptend=(THETADR-AB_gT)*mk3D_mod(RAC,THETADR)   # degC.m/s * m^2 = degC.m^3/s
    tmptend = tmptend                          # degC.m^3/s

    # redefine all the terms as a list from how we did before
    # if mapping, just return the map of terms as 3D fields
    # redefine all the terms as a list from how we did before
    # if mapping, just return the map of terms as 3D fields
    if mapping:
        termsT3D = {}
        termsT3D["ADVh"] = ADVhT
        termsT3D["ADVr"] = ADVrT
        termsT3D["DFhT"] = DFhT
        termsT3D["DFrT"] = DFrT
        termsT3D["surf"] = Ft_surf
        termsT3D["TFLUX"] = Ft_surf
        termsT3D["zconv_top"] = Ft_zconv_top
        termsT3D["swtop"] = Ft_swtop
        termsT3D["KPP"] = tmpkpp
        termsT3D["tend"] = tmptend
    
        return Msum, termsT3D

    term_names = [
        "ADVh",
        "ADVr",
        "DFhT",
        "DFrT",
        "TFLUX",
        "zconv_top",
        "swtop",
        "KPP",
        "tend",
    ]
    
    term_fields = {
        "ADVh": ADVhT,
        "ADVr": ADVrT,
        "DFhT": DFhT,
        "DFrT": DFrT,
        "TFLUX": Ft_surf,
        "zconv_top": Ft_zconv_top,
        "swtop": Ft_swtop,
        "KPP": tmpkpp,
        "tend": tmptend,
    }
    
    nterms = len(term_names)
    dF_Tnew = np.zeros((nterms, nT - 1))
    G_T_offline_new = np.zeros((nterms, nT - 1))
    Lijnew = np.zeros((1, nT - 1), dtype=int)
    
    T_flat = np.ravel(THETA * mymsk3d, order='F')
    term_flats = {
        name: np.ravel(field * mymsk3d, order='F')
        for name, field in term_fields.items()
    }
    
    for i in range(nT - 1):
        ij = np.where((T_flat >= binmidT[i]) & (T_flat < binmidT[i + 1]))[0]
        Lijnew[0, i] = len(ij)
    
        if len(ij) > 0:
            for it, name in enumerate(term_names):
                dF_Tnew[it, i] = np.nansum(term_flats[name][ij])
    
    G_T_offline_new = dF_Tnew / binwidthT1[None, :]
    
    dF_T_dict = {name: dF_Tnew[it] for it, name in enumerate(term_names)}
    G_T_dict  = {name: G_T_offline_new[it] for it, name in enumerate(term_names)}

    return Msum, dF_T_dict

def create_layers_totalSALT(tsstr,mygrid,myparms,dirdiags,dirstate,layers_path,mymsk,nz,ny,nx,nfx,nfy,dt,boundsT,boundsS,mapping=False,debug=False):
    # do the same copying over but for SALT terms (from the original verification on 12/15)
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
    binsTH_edges = boundsT.reshape(boundsT.shape[0])
    binsTH_centers = (binsTH_edges[:-1] + binsTH_edges[1:])/2
    nT = binsTH_edges.shape[0]-1
    
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
    if debug:
        mymsk = np.full((ny,nx),np.nan)
        mymsk[150,19] = 1
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

    # do this manually from ADVh
    nT   = boundsT.size - 1
    nTm1 = nT - 1
    nS = boundsS.size -1 
    nSm1 = nS - 1
    theta_flat = THETA.ravel()
    salt_flat = SALT.ravel()
    # binmidT[i] <= THETA < binmidT[i+1], i=0..nT-2
    bin_idx_mid = np.digitize(theta_flat, binmidT, right=False) - 1
    valid_mid   = (bin_idx_mid >= 0) & (bin_idx_mid < nTm1) & np.isfinite(theta_flat)
    idx_mid     = bin_idx_mid[valid_mid]
    bin_idx_midS = np.digitize(salt_flat, binmidS, right=False) - 1
    valid_midS  = (bin_idx_midS >= 0) & (bin_idx_midS < nSm1) & np.isfinite(salt_flat)
    idx_midS     = bin_idx_midS[valid_midS]
    
    ############################################################
    # get the internal transformations
    
    file_name = "budg3d_hflux_set2"
    meta_budg3d_hflux_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_hflux_set2["fldList"])
    varnames = np.array(["ADVx_SLT","ADVy_SLT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    ADVx_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    ADVy_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    
    # now 3d zfluxes
    file_name = "budg3d_zflux_set2"
    meta_budg3d_zflux_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_zflux_set1["fldList"])
    varnames = np.array(["ADVr_SLT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    ADVr_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    ADVr_SLT = ADVr_SLT.reshape(nz,ny,nx)
    
    hf = mygrid['hFacC']
    
    ADV_hconv = calc_UV_conv_mod(nfx, nfy,get_aste_faces(ADVx_SLT.reshape(nz, ny, nx), nfx, nfy),get_aste_faces(ADVy_SLT.reshape(nz, ny, nx), nfx, nfy))
    ADV_hconv = ADV_hconv * hf   # degC·m^3/s at cell centers (matches: ff.DFh = ff.DFh .* hf)
    ADVhS = ADV_hconv
    
    trWtopADV = -(ADVr_SLT)
    
    ADVrS = np.zeros((nz,ny,nx),dtype=float)
    ADVrS[:-1,:,:] = (trWtopADV[:-1] - trWtopADV[1:]) 

    # --- reshape to 3D ---
    ADVx_SLT = ADVx_SLT.reshape((nz, ny, nx))   # advective salt flux on x-faces (PSU·m^3/s)
    ADVy_SLT = ADVy_SLT.reshape((nz, ny, nx))   # advective salt flux on y-faces (PSU·m^3/s)
    THETA = THETA.reshape((nz,ny,nx))
    SALT     = SALT.reshape((nz, ny, nx))       # cell-centered salinity (PSU)
    
    # ------------------------------------------------------------------
    # 1) Build SALT at faces
    # ------------------------------------------------------------------
    
    # x-faces: between (i-1, i) along x
    salt_x = np.zeros_like(ADVx_SLT)
    salt_x[:, :, 1:] = 0.5 * (SALT[:, :, 1:] + SALT[:, :, :-1])
    salt_x[:, :, 0]  = salt_x[:, :, 1]      # simple fill for western boundary
    salt_y = np.zeros_like(ADVy_SLT)
    salt_y[:, 1:, :] = 0.5 * (SALT[:, 1:, :] + SALT[:, :-1, :])
    salt_y[:, 0, :]  = salt_y[:, 1, :]      # simple fill for southern boundary

    # x-faces: between (i-1, i) along x
    theta_x = np.zeros_like(ADVx_SLT)
    theta_x[:, :, 1:] = 0.5 * (THETA[:, :, 1:] + THETA[:, :, :-1])
    theta_x[:, :, 0]  = theta_x[:, :, 1]   # simple fill for western boundary
    theta_y = np.zeros_like(ADVy_SLT)
    theta_y[:, 1:, :] = 0.5 * (THETA[:, 1:, :] + THETA[:, :-1, :])
    theta_y[:, 0, :]  = theta_y[:, 1, :]   # simple fill for southern boundary
    
    # ------------------------------------------------------------------
    # 2) Convert salt flux (PSU·m^3/s) -> volume flux (m^3/s) if desired
    #    q_vol = q_salt / salt_face
    #    (kept as raw SALT flux here, matching your THETA snippet)
    # ------------------------------------------------------------------
    
    eps = 1e-6  # avoid divide-by-zero if you later enable division
    
    ADVx_vol = np.zeros_like(ADVx_SLT)
    mask_x   = np.isfinite(salt_x) & (np.abs(salt_x) > eps)
    ADVx_vol[mask_x] = ADVx_SLT[mask_x]  # / salt_x[mask_x]
    
    ADVy_vol = np.zeros_like(ADVy_SLT)
    mask_y   = np.isfinite(salt_y) & (np.abs(salt_y) > eps)
    ADVy_vol[mask_y] = ADVy_SLT[mask_y]  # / salt_y[mask_y]
    
    # bolus (kept identical pattern to your THETA snippet)
    ADVx_vol = np.zeros_like(ADVx_SLT)
    mask_x   = np.isfinite(salt_x) & (np.abs(salt_x) > eps)
    ADVx_vol[mask_x] = ADVx_SLT[mask_x]  # / salt_x[mask_x]
    
    ADVy_vol = np.zeros_like(ADVy_SLT)
    mask_y   = np.isfinite(salt_y) & (np.abs(salt_y) > eps)
    ADVy_vol[mask_y] = ADVy_SLT[mask_y]  # / salt_y[mask_y]
    
    # ------------------------------------------------------------
    # Build per-gate (salt_face, flux) samples and bin -> arrays are ZERO-filled
    # ------------------------------------------------------------
    # NOTE: replace `bin_gate_by_face_theta_zero` with your salinity-bin function name
    #       if it’s different; arguments assumed: (tracer_vals, flux_vals, edges, nBinsMinus1)
    
    # ---- BSO ----
    theta_BSO_list, salt_BSO_list, flux_BSO_list = [], [], []
    
    for j, i in zip(y_bsoh, x_bsoh):
        theta_BSO_list.append(theta_x[:, j, i].ravel())
        salt_BSO_list.append(salt_x[:, j, i].ravel())
        flux_BSO_list.append(ADVx_vol[:, j, i].ravel())

    for j, i in zip(y_bsov, x_bsov):
        theta_BSO_list.append(theta_y[:, j, i].ravel())
        salt_BSO_list.append(salt_y[:, j, i].ravel())
        flux_BSO_list.append((-ADVy_vol[:, j, i]).ravel())

    theta_BSO = np.concatenate(theta_BSO_list) if theta_BSO_list else np.array([], dtype=float)
    salt_BSO  = np.concatenate(salt_BSO_list)  if salt_BSO_list  else np.array([], dtype=float)
    flux_BSO  = np.concatenate(flux_BSO_list)  if flux_BSO_list  else np.array([], dtype=float)

    # build the M term to plot against T
    ADVh_BSO = bin_gate_by_face_theta_zero(salt_BSO, flux_BSO, binmidS, nSm1)
    G_BSO    = ADVh_BSO / binwidthS1

    # build the T--S version of this
    ADVh_BSO_TS = bin_gate_by_face_TS_zero(theta_BSO, salt_BSO, flux_BSO,binmidT, binmidS, nTm1, nSm1)
    G_BSO_TS = ADVh_BSO_TS / (binwidthT1[:, None] * binwidthS1[None, :])
    
    
    # ---- FJNZ ----
    theta_FJNZ_list, salt_FJNZ_list, flux_FJNZ_list = [], [], []

    for j, i in zip(y_fjnzv, x_fjnzv):
        theta_FJNZ_list.append(theta_x[:, j, i].ravel())
        salt_FJNZ_list.append(salt_x[:, j, i].ravel())
        flux_FJNZ_list.append((-ADVx_vol[:, j, i]).ravel())

    theta_FJNZ = np.concatenate(theta_FJNZ_list) if theta_FJNZ_list else np.array([], dtype=float)
    salt_FJNZ = np.concatenate(salt_FJNZ_list) if salt_FJNZ_list else np.array([], dtype=float)
    flux_FJNZ = np.concatenate(flux_FJNZ_list) if flux_FJNZ_list else np.array([], dtype=float)

    # build the G term to plot against S
    ADVh_FJNZ = bin_gate_by_face_theta_zero(salt_FJNZ, flux_FJNZ, binmidS, nSm1)
    G_FJNZ    = ADVh_FJNZ / binwidthS1

    # and build the TS version
    ADVh_FJNZ_TS = bin_gate_by_face_TS_zero(theta_FJNZ, salt_FJNZ, flux_FJNZ,binmidT, binmidS, nTm1, nSm1)
    G_FJNZ_TS = ADVh_FJNZ_TS / (binwidthT1[:, None] * binwidthS1[None, :])
    
    # ---- SPFJ ----
    theta_SPFJ_list, salt_SPFJ_list, flux_SPFJ_list = [], [], []
    
    for j, i in zip(y_spfjv2, x_spfjv2):
        theta_SPFJ_list.append(theta_y[:, j, i].ravel())
        salt_SPFJ_list.append(salt_y[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVy_vol[:, j, i]).ravel())
    
    for j, i in zip(y_spfjh2, x_spfjh2):
        theta_SPFJ_list.append(theta_x[:, j, i].ravel())
        salt_SPFJ_list.append(salt_x[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVx_vol[:, j, i]).ravel())
    
    for j, i in zip(y_spfjb2, x_spfjb2):
        theta_SPFJ_list.append(theta_x[:, j, i].ravel())
        salt_SPFJ_list.append(salt_x[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVx_vol[:, j, i]).ravel())
    
        theta_SPFJ_list.append(theta_y[:, j, i].ravel())
        salt_SPFJ_list.append(salt_y[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVy_vol[:, j, i]).ravel())

    theta_SPFJ = np.concatenate(theta_SPFJ_list) if theta_SPFJ_list else np.array([], dtype=float)
    salt_SPFJ = np.concatenate(salt_SPFJ_list) if salt_SPFJ_list else np.array([], dtype=float)
    flux_SPFJ = np.concatenate(flux_SPFJ_list) if flux_SPFJ_list else np.array([], dtype=float)

    # build the M term to plot against S
    ADVh_SPFJ = bin_gate_by_face_theta_zero(salt_SPFJ, flux_SPFJ, binmidS, nSm1)
    G_SPFJ    = ADVh_SPFJ / binwidthS1

    # build the TS diagram from this
    ADVh_SPFJ_TS = bin_gate_by_face_TS_zero(theta_SPFJ, salt_SPFJ, flux_SPFJ,binmidT, binmidS, nTm1, nSm1)
    G_SPFJ_TS = ADVh_SPFJ_TS / (binwidthT1[:, None] * binwidthS1[None, :])
        
    # ---- NZRU ----
    theta_NZRU_list, salt_NZRU_list, flux_NZRU_list = [], [], []
    
    for j, i in zip(y_nzruv, x_nzruv):
        theta_NZRU_list.append(theta_x[:, j, i].ravel())
        salt_NZRU_list.append(salt_x[:, j, i].ravel())
        flux_NZRU_list.append((-ADVx_vol[:, j, i]).ravel())

    theta_NZRU = np.concatenate(theta_NZRU_list) if theta_NZRU_list else np.array([], dtype=float)
    salt_NZRU = np.concatenate(salt_NZRU_list) if salt_NZRU_list else np.array([], dtype=float)
    flux_NZRU = np.concatenate(flux_NZRU_list) if flux_NZRU_list else np.array([], dtype=float)

    # create the volume transformation to plot against S
    ADVh_NZRU = bin_gate_by_face_theta_zero(salt_NZRU, flux_NZRU, binmidS, nSm1)
    G_NZRU    = ADVh_NZRU / binwidthS1

    # create the TS diagram for this
    ADVh_NZRU_TS = bin_gate_by_face_TS_zero(theta_NZRU, salt_NZRU, flux_NZRU,binmidT, binmidS, nTm1, nSm1)
    G_NZRU_TS = ADVh_NZRU_TS / (binwidthT1[:, None] * binwidthS1[None, :])
    
    
    # instead of returning Msum here, make Msum a dictionary with these four gates
    Msum = {}
    Msum["BSO"]  = ADVh_BSO
    Msum["FJNZ"] = ADVh_FJNZ
    Msum["SPFJ"] = ADVh_SPFJ
    Msum["NZRU"] = ADVh_NZRU
    Msum["Msum"] = ADVh_BSO + ADVh_FJNZ + ADVh_SPFJ + ADVh_NZRU

    # add the not-normalized versions of this
    Msum["BSO_TSnn"] = ADVh_BSO_TS
    Msum["FJNZ_TSnn"] = ADVh_FJNZ_TS
    Msum["SPFJ_TSnn"] = ADVh_SPFJ_TS
    Msum["NZRU_TSnn"] = ADVh_NZRU_TS  # we can run a validation with these first and then see
    Msum["Msum_TSnn"] = ADVh_BSO_TS + ADVh_FJNZ_TS + ADVh_SPFJ_TS + ADVh_NZRU_TS  # PSU.m^3/s

    # add the normalized versions of this to get the vector
    Msum["BSO_TS"] = G_BSO_TS
    Msum["FJNZ_TS"] = G_FJNZ_TS
    Msum["SPFJ_TS"] = G_SPFJ_TS
    Msum["NZRU_TS"] = G_NZRU_TS
    Msum["Msum_TS"] = G_BSO_TS + G_FJNZ_TS + G_SPFJ_TS + G_NZRU_TS  # m^3/s/degC


    ########################################################################################################################
    
    ## do the advective convergence
    file_name = "budg3d_hflux_set2"
    meta_budg3d_hflux_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_hflux_set2["fldList"])
    varnames = np.array(["DFxE_SLT","DFyE_SLT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    DFxE_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    DFyE_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    
    
    # now 3d zfluxes
    file_name = "budg3d_zflux_set2"
    meta_budg3d_zflux_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_zflux_set2["fldList"])
    varnames = np.array(["DFrE_SLT","DFrI_SLT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    DFrE_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    DFrI_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    DFrE_SLT = DFrE_SLT.reshape(nz,ny,nx)
    DFrI_SLT = DFrI_SLT.reshape(nz,ny,nx)
    
    DF_hconv = calc_UV_conv_mod(nfx, nfy,get_aste_faces(DFxE_SLT.reshape(nz, ny, nx), nfx, nfy),get_aste_faces(DFyE_SLT.reshape(nz, ny, nx), nfx, nfy))
    DF_hconv = DF_hconv * hf   # degC·m^3/s at cell centers (matches: ff.DFh = ff.DFh .* hf)
    DFhS = DF_hconv
    
    trWtopDF = -(DFrE_SLT+DFrI_SLT)
    
    DFrS = np.zeros((nz,ny,nx),dtype=float)
    DFrS[:-1,:,:] = (trWtopDF[:-1] - trWtopDF[1:])
    
    # load the surface terms
    # read fluxes
    file_name = 'budg2d_zflux_set1'
    meta_budg2d_zflux_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg2d_zflux_set1["fldList"])
    varnames = np.array(["oceSPflx","SFLUX"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    oceSPflx,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    SFLUX,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    oceSPflx = oceSPflx.reshape(ny,nx)
    SFLUX = SFLUX.reshape(ny,nx)
    
    # read relax and salt mass
    file_name = "budg2d_zflux_set2"
    meta_budg2d_zflux_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg2d_zflux_set2["fldList"])
    varnames = np.array(["oceSflux","WSLTMASS"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        if len(irec[0]) > 0:
            recs = np.append(recs, irec[0][0])
    oceSflux,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    WSLTMASS,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    oceSflux = oceSflux.reshape(ny,nx)
    WSLTMASS = WSLTMASS.reshape(ny,nx)
    
    # read kpp tend and from 3d zflux
    file_name = "budg3d_kpptend_set1"
    meta_budg3d_kpptend_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_kpptend_set1["fldList"])
    varnames = np.array(["oceSPtnd","KPPg_SLT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        if len(irec[0]) > 0:
            recs = np.append(recs, irec[0][0])
    oceSPtnd,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    KPPg_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    oceSPtnd = oceSPtnd.reshape(nz,ny,nx)
    KPPg_SLT = KPPg_SLT.reshape(nz,ny,nx)
    
    RAC3 = np.tile(RAC[np.newaxis,:,:],(nz,1,1))
    file_name = "budg3d_kpptend_set1"
    meta_budg3d_kpptend_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_kpptend_set1["fldList"])
    
    rec_oceSPtnd = np.where(fldlist == "oceSPtnd")[0][0]
    
    oceSPtnd, its, meta = rdmds(
        os.path.join(dirdiags, file_name),
        t2,
        returnmeta=True,
        rec=rec_oceSPtnd
    )
    oceSPtnd = oceSPtnd.reshape(nz, ny, nx)
    
    RAC3 = np.tile(RAC[np.newaxis, :, :], (nz, 1, 1))
    
    # 3D redistributed salt surface term
    sptop = mk3D_mod(oceSPflx, oceSPtnd) - np.cumsum(oceSPtnd, axis=0)
    sptop = sptop * RAC3        # g/s
    
    # 2D top boundary term
    zconv_top_salt = (SFLUX + oceSPflx) * RAC   # g/s
    
    
    def surface_contrib_JS(zconv_top_salt, sptop, rho, fill_last=0.0):
        """
        zconv_top_salt : (ny, nx)     top boundary salt term in g/s
        sptop          : (nz, ny, nx) redistributed salt flux profile in g/s
        returns:
          JsurfS       : (nz, ny, nx) in PSU.m^3/s
        """
        nz_, ny_, nx_ = sptop.shape
        eS = zconv_top_salt.reshape(1, ny_, nx_)
        J = np.empty_like(sptop, dtype=float)
    
        J[0] = (eS[0] - sptop[1]) / rho
        J[1:nz_] = -(sptop[1:nz_] - sptop[0:nz_-1]) / rho
        J[-1] = fill_last
    
        return J
    
    
    # original full surface salt contribution
    Fs_surf = surface_contrib_JS(
        zconv_top_salt,
        sptop,
        myparms['rhoconst']
    )
    
    # split into two terms
    Fs_zconv_top = surface_contrib_JS(
        zconv_top_salt,
        np.zeros((nz, ny, nx)),
        myparms['rhoconst']
    )
    
    Fs_sptop = surface_contrib_JS(
        np.zeros((ny, nx)),
        sptop,
        myparms['rhoconst']
    )
    
    # do the vertical convergence for KPP
    trWtopKPP = -(KPPg_SLT)         # PSU.m^3/s
    
    tmpkpp = np.full((nz, ny, nx), np.nan)
    tmpkpp[:-1, :, :] = trWtopKPP[:-1] - trWtopKPP[1:]
    
    
    file_name = 'budg3d_snap_set2'
    meta_budg3d_snap_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_snap_set2["fldList"])
    varnames = np.array(["SALTDR"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    
    SALTDR = np.full((len(tsstr), nz, ny, nx), np.nan)
    for i in range(len(tsstr)):
        thisSALTDR, its, meta = rdmds(
            os.path.join(dirdiags, file_name),
            int(tsstr[i]),
            returnmeta=True,
            rec=recs[0]
        )
        thisSALTDR = thisSALTDR.reshape(nz, ny, nx)
        SALTDR[i] = thisSALTDR
    
    SALTDR = (SALTDR[1, :, :, :] - SALTDR[0, :, :, :]) / dt    # PSU.m/s
    tmptend = (SALTDR - 0) * mk3D_mod(RAC, SALTDR)              # PSU.m^3/s
    
    
    # if mapping, just return the map of terms as 3D fields
    if mapping:
        termsS3D = {}
        termsS3D["ADVh"] = ADVhS
        termsS3D["ADVr"] = ADVrS
        termsS3D["DFhS"] = DFhS
        termsS3D["DFrS"] = DFrS
        termsS3D["surf"] = Fs_surf
        termsS3D["zconv_top"] = Fs_zconv_top
        termsS3D["sptop"] = Fs_sptop
        termsS3D["KPP"] = tmpkpp
        termsS3D["tend"] = tmptend
    
        return Msum, termsS3D
    
    
    # binning with full surface breakdown
    term_names = [
        "ADVh",
        "ADVr",
        "DFhS",
        "DFrS",
        "surf",
        "zconv_top",
        "sptop",
        "KPP",
        "tend",
    ]
    
    term_fields = {
        "ADVh": ADVhS,
        "ADVr": ADVrS,
        "DFhS": DFhS,
        "DFrS": DFrS,
        "surf": Fs_surf,
        "zconv_top": Fs_zconv_top,
        "sptop": Fs_sptop,
        "KPP": tmpkpp,
        "tend": tmptend,
    }
    
    nterms = len(term_names)
    dF_Snew = np.zeros((nterms, nS - 1))
    G_S_offline_new = np.zeros((nterms, nS - 1))
    Lijnew = np.zeros((1, nS - 1), dtype=int)
    
    S_flat = np.ravel(SALT * hf * mymsk3d, order='F')
    term_flats = {
        name: np.ravel(field * hf * mymsk3d, order='F')
        for name, field in term_fields.items()
    }
    
    for i in range(nS - 1):
        ij = np.where((S_flat >= binmidS[i]) & (S_flat < binmidS[i + 1]))[0]
        Lijnew[0, i] = len(ij)
    
        if len(ij) > 0:
            for it, name in enumerate(term_names):
                dF_Snew[it, i] = np.nansum(term_flats[name][ij])
    
    G_S_offline_new = dF_Snew / binwidthS1[None, :]
    
    dF_S_dict = {name: dF_Snew[it] for it, name in enumerate(term_names)}
    G_S_dict  = {name: G_S_offline_new[it] for it, name in enumerate(term_names)}
    
    return Msum, dF_S_dict


def create_TS_mesh(binned_theta, binned_salinity, attr, idxs, nT, nS, dT, dS):
    """
    Bin attr into T-S space.

    Inputs
    ------
    binned_theta : array
        Bin indices for theta, shape (nz, ny, nx) or (time, nz, ny, nx)
    binned_salinity : array
        Bin indices for salinity, same shape as binned_theta
    attr : array
        Attribute to sum, same shape as binned arrays
    idxs : tuple
        np.where(mask == 1), i.e. basin horizontal indices
    nT, nS : int
        Number of T and S bins
    dT, dS : float or 1D arrays
        Bin widths in T and S

    Returns
    -------
    mesh : array
        Shape (nT, nS), volume summed into T-S bins and normalized by dT*dS
    """

    # extract basin points
    if attr.ndim == 4:
        raise ValueError("Pass a single time slice, not a time-dependent 4D array.")
    elif attr.ndim == 3:
        thisattr = attr[:, idxs[0], idxs[1]]              # (nz, npoints)
        thisT    = binned_theta[:, idxs[0], idxs[1]]      # (nz, npoints)
        thisS    = binned_salinity[:, idxs[0], idxs[1]]   # (nz, npoints)
    elif attr.ndim == 2:
        thisattr = attr[idxs[0], idxs[1]]
        thisT    = binned_theta[idxs[0], idxs[1]]
        thisS    = binned_salinity[idxs[0], idxs[1]]
    else:
        raise ValueError("attr must be 2D or 3D")

    # flatten
    thisattr = thisattr.ravel()
    thisT    = thisT.ravel()
    thisS    = thisS.ravel()

    # keep only valid bin assignments
    valid = (
        np.isfinite(thisattr) &
        np.isfinite(thisT) &
        np.isfinite(thisS) &
        (thisT >= 0) & (thisT < nT) &
        (thisS >= 0) & (thisS < nS)
    )

    thisattr = thisattr[valid]
    thisT    = thisT[valid].astype(int)
    thisS    = thisS[valid].astype(int)

    # build mesh: shape (nT, nS)
    mesh = np.zeros((nT, nS))
    np.add.at(mesh, (thisT, thisS), thisattr)

    # normalize by bin area in T-S space
    # if dT and dS are scalars:
    if np.ndim(dT) == 0 and np.ndim(dS) == 0:
        mesh = mesh / (dT * dS)
    else:
        # if dT, dS are 1D arrays of bin widths
        mesh = mesh / (np.asarray(dT)[:, None] * np.asarray(dS)[None, :])

    return mesh

def bin_3d_term_to_TS(term3d, THETA, SALT, mymsk3d, binmidT, binmidS, binwidthT1, binwidthS1):
    nT = len(binmidT)
    nS = len(binmidS)

    dF_TS = np.zeros((nT-1, nS-1), dtype=float)
    L_TS  = np.zeros((nT-1, nS-1), dtype=int)

    # Flatten
    T_flat = np.ravel(THETA * mymsk3d, order="F")
    S_flat = np.ravel(SALT  * mymsk3d, order="F")
    X_flat = np.ravel(term3d * mymsk3d, order="F")

    # Bin indices
    iT = np.searchsorted(binmidT, T_flat, side="right") - 1
    iS = np.searchsorted(binmidS, S_flat, side="right") - 1

    # Valid points
    ok = (iT >= 0) & (iT < nT-1) & (iS >= 0) & (iS < nS-1) & np.isfinite(X_flat)
    iT = iT[ok]
    iS = iS[ok]
    vals = X_flat[ok]

    # Accumulate
    np.add.at(dF_TS, (iT, iS), vals)
    np.add.at(L_TS,  (iT, iS), 1)

    # Normalize by both bin widths
    dA = binwidthT1[:, None] * binwidthS1[None, :]  # An suggests doing this separately in future
                                                    # first division in dT or dS gets you G_T or G_S
                                                    # second division by dS or dT gets you the J component of J_S or J_T
    G_TS = dF_TS / dA                               # should not be called G_TS here!

    return G_TS, dF_TS, L_TS

def get_div(Jy, Jx, dT_centers, dS_centers):
    """
    inputs:
        Jy: transport in T direction, shape (nT-1, nS-1), units m^3/s/PSU
            defined on center of T_centers,S_centers
        Jx: transport in S direction, shape (nT-1, nS-1), units m^3/s/degC
            defined on center of T_centers,S_centers
        dT_edges: T-bin widths at interior T intervals, shape (nT-1,) or compatible
        dS_edges: S-bin widths at interior S intervals, shape (nS-1,) or compatible

    outputs:
        gradJ: divergence on interior cell centers, shape (nT-2, nS-2),
               units Sv/PSU/degC
    """

    # Interpolate fluxes onto the INNER faces of the interior cells first
    # Jy_inner: top/bottom faces for interior cells, remove outer S columns
    Jy_inner = 0.5 * (Jy[1:, :] + Jy[:-1, :])   # shape (nT-2, nS-1)   # bring these up to the inner face  (in temp)

    # Jx_inner: left/right faces for interior cells, remove outer T rows
    Jx_inner = 0.5 * (Jx[:, 1:] + Jx[:, :-1])   # shape (nT-2, nS-1)   # bring these to the inner faces LR (in salt)

    # Then take differences across those inner faces
    # check this is the correct sign for + in
    dJy = Jy_inner[1:, :] - Jy_inner[:-1, :]    # shape (nT-3, nS-2)
    dJx = Jx_inner[:, 1:] - Jx_inner[:, :-1]    # shape (nT-2, nS-3)

    # Divide by bin widths
    dJydT = dJy / dT_centers[1:-1, None]
    dJxdS = dJx / dS_centers[None, 1:-1]

    
    #print()

    #print(np.where(dJydT!=0),np.where(dJxdS!=0))

    # Negative so positive means creation
    gradJ = -(dJydT[:,1:-1] + dJxdS[1:-1,:])

    return gradJ * 1e-6   # Sv/PSU/degC