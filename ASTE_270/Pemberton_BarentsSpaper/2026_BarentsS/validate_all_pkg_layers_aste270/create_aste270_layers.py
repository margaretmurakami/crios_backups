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


# Summary of key functions in this file:
#
# create_layersTHETA:
#   - Computes temperature (heat) budget terms in physical space (xyz)
#     including ADVh, ADVr, DFh, DFr, surface forcing, KPP, and tendency
#   - Bins these terms into temperature space (T) to produce dF_Tnew (m^3/s, degC·m^3/s)
#   - Computes gateway transports (BSO, FJNZ, SPFJ, NZRU) using face-centered tracers
#   - Returns:
#       Msum        → gateway contributions (1D T bins and 2D T-S distributions)
#       termsT3D    → full 3D budget terms in physical space
#       dF_Tnew     → binned temperature budget contributions
#
# create_layersSALT:
#   - Same structure as create_layersTHETA but for salinity
#   - Computes salt budget terms in physical space (xyz)
#   - Bins into salinity space (S) to produce dF_Snew (m^3/s, PSU·m^3/s)
#   - Computes gateway transports and T-S distributions
#   - Returns:
#       Msum        → gateway contributions (1D S bins and 2D T-S distributions)
#       termsS3D    → full 3D budget terms in physical space
#       dF_Snew     → binned salinity budget contributions
#
# Helper functions:
#   - bin_3d_term_to_TS: bins 3D fields into T-S space
#   - bin_gate_by_face_theta_zero / bin_gate_by_face_TS_zero:
#         bin gateway fluxes into T or T-S space using face tracers
#   - get_div: computes divergence in T-S space from flux components
#   - create_TS_mesh: builds T-S distributions from gridded fields
#
# Overall:
#   This file connects MITgcm diagnostics → physical-space budgets → T/S and T-S
#   layer-space representations, including gateway decomposition.


# for creating the J vectors, we need to bin to both T and S
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


# for creating a volume mesh for THETA and SALT for an individual time step
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

# copy over the THETA calculation from adv_closure_TS
# create the total tendency first
# from create_layers import create_layersTHETA,create_layersSALT
def create_layersTHETA(tsstr,mygrid,myparms,dirdiags,dirState,layers_path,mymsk,nz,ny,nx,nfx,nfy,dt,mapping=False):
    # we want to create dF_Tnew, basically, which contains the information from the layers output mimicked by ASTER1
    # let's just check with ADVh first
    mymsk3d = np.tile(mymsk[np.newaxis,:,:],(nz,1,1))
    t2 = int(tsstr[1])
    hf = mygrid['hFacC']
    RAC = mygrid['RAC']
    
    # load THETA
    file_name = "state_3d_set1"
    meta_state_3d_set1 = parsemeta(dirState + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_state_3d_set1["fldList"])
    varnames = np.array(["THETA","SALT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    THETA,its,meta = rdmds(os.path.join(dirState, file_name),t2,returnmeta=True,rec=recs[0])
    SALT,its,meta = rdmds(os.path.join(dirState, file_name),t2,returnmeta=True,rec=recs[1])
    THETA = THETA.reshape(nz,ny,nx)
    SALT = SALT.reshape(nz,ny,nx)

    # LOAD ADV FOR BOTH T AND S
    file_name = "budg3d_hflux_set2"
    meta_budg3d_hflux_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_hflux_set2["fldList"])
    varnames = np.array(["ADVx_TH","ADVy_TH","ADVx_SLT","ADVy_SLT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    ADVx_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    ADVy_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    
    # for temp, get convergence
    ADV_hconvT = calc_UV_conv_mod(nfx, nfy,get_aste_faces(ADVx_TH.reshape(nz, ny, nx), nfx, nfy),get_aste_faces(ADVy_TH.reshape(nz, ny, nx), nfx, nfy))
    ADV_hconvT = ADV_hconvT   # degC·m^3/s at cell centers (matches: ff.DFh = ff.DFh .* hf)
    ADVhT = ADV_hconvT
    
    # now 3d zfluxes
    file_name = "budg3d_zflux_set2"
    meta_budg3d_zflux_set1 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_zflux_set1["fldList"])
    varnames = np.array(["ADVr_TH","ADVr_SLT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    ADVr_TH,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[0])
    ADVr_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[1])
    ADVr_TH = ADVr_TH.reshape(nz,ny,nx)

    # for temp, get convergence
    trWtopADV = -(ADVr_TH)
    ADVrT = np.zeros((nz,ny,nx),dtype=float)
    ADVrT[:-1,:,:] = (trWtopADV[:-1] - trWtopADV[1:])
    
    ## load the TS bins
    boundsT = np.round(np.squeeze(rdmds(layers_path + "layers2TH")).ravel(),1)
    nT = boundsT.size - 1
    boundsS = np.round(np.squeeze(rdmds(layers_path + "layers1SLT")).ravel(),1)
    nS = boundsS.size - 1
    
    binwidthT = boundsT[1:] - boundsT[:-1]
    binwidthS = boundsS[1:] - boundsS[:-1]
    
    binwidthT1 = 0.5 * (binwidthT[1:] + binwidthT[:-1])
    binwidthS1 = 0.5 * (binwidthS[1:] + binwidthS[:-1])
    
    binmidT = (boundsT[1:] + boundsT[:-1]) /2
    binmidS = (boundsS[1:] + boundsS[:-1]) /2
    
    nT = nT
    nS = nS
    nTm1 = nT-1
    nSm1 = nS-1

    ####################################################################################################
    # pause, create the gates from ADVx and ADVy
    data = np.load("/home/mmurakami/crios_backups/an_helper_functions/gates_BSO.npz")
    y_bsoh, x_bsoh = data["y_bsoh"], data["x_bsoh"]
    y_bsov, x_bsov = data["y_bsov"], data["x_bsov"]
    y_fjnzv, x_fjnzv = data["y_fjnzv"], data["x_fjnzv"]
    y_fjnz,  x_fjnz  = data["y_fjnz"],  data["x_fjnz"]
    y_nzruv, x_nzruv = data["y_nzruv"], data["x_nzruv"]
    y_spfjh, x_spfjh = data["y_spfjh"], data["x_spfjh"]
    y_spfjv, x_spfjv = data["y_spfjv"], data["x_spfjv"]
    y_spfjb, x_spfjb = data["y_spfjb"], data["x_spfjb"]
    
    # --- reshape to 3D ---
    ADVx = ADVx_TH.reshape((nz, ny, nx))   # advective heat flux on x-faces
    ADVy = ADVy_TH.reshape((nz, ny, nx))   # advective heat flux on y-faces
    THETA = THETA.reshape((nz, ny, nx))   # cell-centered temperature
    SALT = SALT.reshape((nz,ny,nx))
    
    # ------------------------------------------------------------------
    # build tracer at faces
    # ------------------------------------------------------------------
    
    # x-faces: between (i-1, i)
    theta_x = np.zeros_like(ADVx, dtype=float)
    theta_x[:, :, 1:] = 0.5 * (THETA[:, :, 1:] + THETA[:, :, :-1])
    theta_x[:, :, 0]  = theta_x[:, :, 1]
    
    # y-faces: between (j-1, j)
    theta_y = np.zeros_like(ADVy, dtype=float)
    theta_y[:, 1:, :] = 0.5 * (THETA[:, 1:, :] + THETA[:, :-1, :])
    theta_y[:, 0, :]  = theta_y[:, 1, :]

    salt_x = np.zeros_like(ADVx, dtype=float)
    salt_x[:, :, 1:] = 0.5 * (SALT[:, :, 1:] + SALT[:, :, :-1])
    salt_x[:, :, 0]  = salt_x[:, :, 1]
    
    salt_y = np.zeros_like(ADVy, dtype=float)
    salt_y[:, 1:, :] = 0.5 * (SALT[:, 1:, :] + SALT[:, :-1, :])
    salt_y[:, 0, :]  = salt_y[:, 1, :]
    
    # ------------------------------------------------------------------
    # BSO: collect face tracer and face flux directly
    # ------------------------------------------------------------------
    theta_BSO_list = []
    salt_BSO_list = []
    flux_BSO_list   = []
    
    # horizontal faces (u-faces)
    for j, i in zip(y_bsoh, x_bsoh):
        theta_BSO_list.append(theta_x[:, j, i].ravel())
        salt_BSO_list.append(salt_x[:, j, i].ravel())
        flux_BSO_list.append(ADVx[:, j, i].ravel())     # + into basin
    
    # vertical faces (v-faces)
    for j, i in zip(y_bsov, x_bsov):
        theta_BSO_list.append(theta_y[:, j, i].ravel())
        salt_BSO_list.append(salt_y[:, j, i].ravel())
        flux_BSO_list.append((-ADVy[:, j, i]).ravel())  # + into basin
    
    theta_BSO = np.concatenate(theta_BSO_list) if theta_BSO_list else np.array([], dtype=float)
    salt_BSO = np.concatenate(salt_BSO_list) if salt_BSO_list else np.array([], dtype=float)
    flux_BSO   = np.concatenate(flux_BSO_list)   if flux_BSO_list   else np.array([], dtype=float)
    
    ADVh_BSO    = bin_gate_by_face_theta_zero(theta_BSO, flux_BSO, binmidT, nTm1)
    ADVh_BSO_TS = bin_gate_by_face_TS_zero(theta_BSO, salt_BSO, flux_BSO, binmidT, binmidS, nTm1, nSm1)
    G_BSO_TS    = ADVh_BSO_TS / (binwidthT1[:, None] * binwidthS1[None, :])
    
    # ------------------------------------------------------------------
    # FJNZ
    # use the same gate convention as your validated version:
    # ADV_FJNZ[:, y_fjnz, x_fjnzv[0]-1] = -ADVx[:, y_fjnz, x_fjnzv[0]]
    # so bin by tracer at the actual entry face (y_fjnz, x_fjnzv[0])
    # ------------------------------------------------------------------
    theta_FJNZ_list = []
    salt_FJNZ_list = []
    flux_FJNZ_list   = []
    
    for j in y_fjnz:
        theta_FJNZ_list.append(theta_x[:, j, x_fjnzv[0]].ravel())
        salt_FJNZ_list.append(salt_x[:, j, x_fjnzv[0]].ravel())
        flux_FJNZ_list.append((-ADVx[:, j, x_fjnzv[0]]).ravel())
    
    theta_FJNZ = np.concatenate(theta_FJNZ_list) if theta_FJNZ_list else np.array([], dtype=float)
    salt_FJNZ = np.concatenate(salt_FJNZ_list) if salt_FJNZ_list else np.array([], dtype=float)
    flux_FJNZ   = np.concatenate(flux_FJNZ_list)   if flux_FJNZ_list   else np.array([], dtype=float)
    
    ADVh_FJNZ = bin_gate_by_face_theta_zero(theta_FJNZ, flux_FJNZ, binmidT, nTm1)
    ADVh_FJNZ_TS = bin_gate_by_face_TS_zero(theta_FJNZ, salt_FJNZ, flux_FJNZ, binmidT, binmidS, nTm1, nSm1)
    G_FJNZ_TS    = ADVh_FJNZ_TS  / (binwidthT1[:, None] * binwidthS1[None, :])
    
    # ------------------------------------------------------------------
    # SPFJ
    # same sign/index conventions as your validated version
    # ------------------------------------------------------------------
    theta_SPFJ_list = []
    salt_SPFJ_list = []
    flux_SPFJ_list   = []
    
    # ADV_SPFJ[:,y_spfjv,x_spfjv-1] -= ADVx[:, y_spfjv, x_spfjv]
    for j, i in zip(y_spfjv, x_spfjv):
        theta_SPFJ_list.append(theta_x[:, j, i].ravel())
        salt_SPFJ_list.append(salt_x[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVx[:, j, i]).ravel())
    
    # ADV_SPFJ[:,y_spfjh-1,x_spfjh] -= ADVy[:, y_spfjh, x_spfjh]
    for j, i in zip(y_spfjh, x_spfjh):
        theta_SPFJ_list.append(theta_y[:, j, i].ravel())
        salt_SPFJ_list.append(salt_y[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVy[:, j, i]).ravel())
    
    # ADV_SPFJ[:,y_spfjb-1,x_spfjb] -= ADVy[:, y_spfjb, x_spfjb]
    for j, i in zip(y_spfjb, x_spfjb):
        theta_SPFJ_list.append(theta_y[:, j, i].ravel())
        salt_SPFJ_list.append(salt_y[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVy[:, j, i]).ravel())
    
    # ADV_SPFJ[:,y_spfjb,x_spfjb-1] -= ADVx[:, y_spfjb, x_spfjb]
    for j, i in zip(y_spfjb, x_spfjb):
        theta_SPFJ_list.append(theta_x[:, j, i].ravel())
        salt_SPFJ_list.append(salt_x[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVx[:, j, i]).ravel())
    
    theta_SPFJ = np.concatenate(theta_SPFJ_list) if theta_SPFJ_list else np.array([], dtype=float)
    salt_SPFJ = np.concatenate(salt_SPFJ_list) if salt_SPFJ_list else np.array([], dtype=float)
    flux_SPFJ   = np.concatenate(flux_SPFJ_list)   if flux_SPFJ_list   else np.array([], dtype=float)
    
    ADVh_SPFJ = bin_gate_by_face_theta_zero(theta_SPFJ, flux_SPFJ, binmidT, nTm1)
    ADVh_SPFJ_TS = bin_gate_by_face_TS_zero(theta_SPFJ, salt_SPFJ, flux_SPFJ, binmidT, binmidS, nTm1, nSm1)
    G_SPFJ_TS    = ADVh_SPFJ_TS  / (binwidthT1[:, None] * binwidthS1[None, :])
    
    # ------------------------------------------------------------------
    # NZRU
    # validated version:
    # ADV_NZRU[:, j, i-1] -= ADVx[:, j, i]
    # so bin by tracer at actual entry x-face (j, i)
    # ------------------------------------------------------------------
    theta_NZRU_list = []
    salt_NZRU_list = []
    flux_NZRU_list   = []
    
    for j, i in zip(y_nzruv, x_nzruv):
        theta_NZRU_list.append(theta_x[:, j, i].ravel())
        salt_NZRU_list.append(salt_x[:, j, i].ravel())
        flux_NZRU_list.append((-ADVx[:, j, i]).ravel())
    
    theta_NZRU = np.concatenate(theta_NZRU_list) if theta_NZRU_list else np.array([], dtype=float)
    salt_NZRU = np.concatenate(salt_NZRU_list) if salt_NZRU_list else np.array([], dtype=float)
    flux_NZRU   = np.concatenate(flux_NZRU_list)   if flux_NZRU_list   else np.array([], dtype=float)
    
    ADVh_NZRU = bin_gate_by_face_theta_zero(theta_NZRU, flux_NZRU, binmidT, nTm1)
    ADVh_NZRU_TS = bin_gate_by_face_TS_zero(theta_NZRU, salt_NZRU, flux_NZRU, binmidT, binmidS, nTm1, nSm1)
    G_NZRU_TS    = ADVh_NZRU_TS  / (binwidthT1[:, None] * binwidthS1[None, :])
    
    Msum = {}
    Msum["BSO"]  = ADVh_BSO
    Msum["FJNZ"] = ADVh_FJNZ
    Msum["SPFJ"] = ADVh_SPFJ
    Msum["NZRU"] = ADVh_NZRU
    Msum["Msum"] = ADVh_BSO + ADVh_FJNZ + ADVh_SPFJ + ADVh_NZRU
    
    Msum["BSO_TSnn"]  = ADVh_BSO_TS
    Msum["FJNZ_TSnn"] = ADVh_FJNZ_TS
    Msum["SPFJ_TSnn"] = ADVh_SPFJ_TS
    Msum["NZRU_TSnn"] = ADVh_NZRU_TS
    Msum["Msum_TSnn"] = ADVh_BSO_TS + ADVh_FJNZ_TS + ADVh_SPFJ_TS + ADVh_NZRU_TS
    
    Msum["BSO_TS"]  = G_BSO_TS
    Msum["FJNZ_TS"] = G_FJNZ_TS
    Msum["SPFJ_TS"] = G_SPFJ_TS
    Msum["NZRU_TS"] = G_NZRU_TS
    Msum["Msum_TS"] = G_BSO_TS + G_FJNZ_TS + G_SPFJ_TS + G_NZRU_TS

    ####################################################################################################

    # create the diffusive term
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
    file_name = "budg3d_zflux_set2"
    meta_budg3d_zflux_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_zflux_set2["fldList"])
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
    
    THETADR =  (THETADR[1, :, :,:] - THETADR[0, :,:, :]) / dt    # degC.m/
    AB_gT = 0
    tmptend=(THETADR-AB_gT)*mk3D_mod(RAC,THETADR)   # degC.m/s * m^2 = degC.m^3/s
    tmptend = tmptend                          # degC.m^3/s

    # create the mapping dictionary here, also we likely don't need to return the normalized G_T_offline
    termsT3D = {}
    termsT3D["ADVh"] = ADVhT
    termsT3D["ADVr"] = ADVrT
    termsT3D["DFhT"] = DFhT
    termsT3D["DFrT"] = DFrT
    termsT3D["surf"] = Ft_surftest
    termsT3D["KPP"] = tmpkpp
    termsT3D["tend"] = tmptend
    

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

    # if mapping is True, just return the closure terms from before as tracer.m^3/s
    if mapping:
        return ADVhT,ADVrT,DFhT,DFrT,Ft_surftest,tmpkpp,tmptend
    
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
    
    return Msum,termsT3D,dF_Tnew  # these will be in units of m^3/s and degC.m^3/s

# great, now we just need to do the same for salt
# manually check create_layersSALT
def create_layersSALT(tsstr,mygrid,myparms,dirdiags,dirState,layers_path,mymsk,nz,ny,nx,nfx,nfy,dt,mapping=False):
    # do the same as previous but return the values in salt
        # we want to create dF_Tnew, basically, which contains the information from the layers output mimicked by ASTER1
    # let's just check with ADVh first
    mymsk3d = np.tile(mymsk[np.newaxis,:,:],(nz,1,1))
    t2 = int(tsstr[1])
    hf = mygrid['hFacC']
    RAC = mygrid['RAC']
    
    # load THETA
    file_name = "state_3d_set1"
    meta_state_3d_set1 = parsemeta(dirState + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_state_3d_set1["fldList"])
    varnames = np.array(["THETA","SALT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    THETA,its,meta = rdmds(os.path.join(dirState, file_name),t2,returnmeta=True,rec=recs[0])
    SALT,its,meta = rdmds(os.path.join(dirState, file_name),t2,returnmeta=True,rec=recs[1])
    THETA = THETA.reshape(nz,ny,nx)
    SALT = SALT.reshape(nz,ny,nx)

    ## load the TS bins
    boundsT = np.round(np.squeeze(rdmds(layers_path + "layers2TH")).ravel(),1)
    nT = boundsT.size - 1
    boundsS = np.round(np.squeeze(rdmds(layers_path + "layers1SLT")).ravel(),1)
    nS = boundsS.size - 1
    
    binwidthT = boundsT[1:] - boundsT[:-1]
    binwidthS = boundsS[1:] - boundsS[:-1]
    
    binwidthT1 = 0.5 * (binwidthT[1:] + binwidthT[:-1])
    binwidthS1 = 0.5 * (binwidthS[1:] + binwidthS[:-1])
    
    binmidT = (boundsT[1:] + boundsT[:-1]) /2
    binmidS = (boundsS[1:] + boundsS[:-1]) /2
    
    nT = nT
    nS = nS
    nTm1 = nT-1
    nSm1 = nS-1

    # load the advective terms for salt
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
    
    ADV_hconv = calc_UV_conv_mod(nfx, nfy,get_aste_faces(ADVx_SLT.reshape(nz, ny, nx), nfx, nfy),get_aste_faces(ADVy_SLT.reshape(nz, ny, nx), nfx, nfy))
    ADV_hconv = ADV_hconv   # PSU·m^3/s at cell centers (matches: ff.DFh = ff.DFh .* hf)
    ADVhS = ADV_hconv
    
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
    
    trWtopADV = -(ADVr_SLT)
    
    ADVrS = np.zeros((nz,ny,nx),dtype=float)
    ADVrS[:-1,:,:] = (trWtopADV[:-1] - trWtopADV[1:])

    ####################################################################################################
    # pause, create the gates from ADVx and ADVy
    data = np.load("/home/mmurakami/crios_backups/an_helper_functions/gates_BSO.npz")
    y_bsoh, x_bsoh = data["y_bsoh"], data["x_bsoh"]
    y_bsov, x_bsov = data["y_bsov"], data["x_bsov"]
    y_fjnzv, x_fjnzv = data["y_fjnzv"], data["x_fjnzv"]
    y_fjnz,  x_fjnz  = data["y_fjnz"],  data["x_fjnz"]
    y_nzruv, x_nzruv = data["y_nzruv"], data["x_nzruv"]
    y_spfjh, x_spfjh = data["y_spfjh"], data["x_spfjh"]
    y_spfjv, x_spfjv = data["y_spfjv"], data["x_spfjv"]
    y_spfjb, x_spfjb = data["y_spfjb"], data["x_spfjb"]
    
    # --- reshape to 3D ---
    ADVx = ADVx_SLT.reshape((nz, ny, nx))   # advective heat flux on x-faces
    ADVy = ADVy_SLT.reshape((nz, ny, nx))   # advective heat flux on y-faces
    THETA = THETA.reshape((nz, ny, nx))   # cell-centered temperature
    SALT = SALT.reshape((nz,ny,nx))
    
    # ------------------------------------------------------------------
    # build tracer at faces
    # ------------------------------------------------------------------
    
    # x-faces: between (i-1, i)
    theta_x = np.zeros_like(ADVx, dtype=float)
    theta_x[:, :, 1:] = 0.5 * (THETA[:, :, 1:] + THETA[:, :, :-1])
    theta_x[:, :, 0]  = theta_x[:, :, 1]
    
    # y-faces: between (j-1, j)
    theta_y = np.zeros_like(ADVy, dtype=float)
    theta_y[:, 1:, :] = 0.5 * (THETA[:, 1:, :] + THETA[:, :-1, :])
    theta_y[:, 0, :]  = theta_y[:, 1, :]

    salt_x = np.zeros_like(ADVx, dtype=float)
    salt_x[:, :, 1:] = 0.5 * (SALT[:, :, 1:] + SALT[:, :, :-1])
    salt_x[:, :, 0]  = salt_x[:, :, 1]
    
    salt_y = np.zeros_like(ADVy, dtype=float)
    salt_y[:, 1:, :] = 0.5 * (SALT[:, 1:, :] + SALT[:, :-1, :])
    salt_y[:, 0, :]  = salt_y[:, 1, :]
    
    # ------------------------------------------------------------------
    # BSO: collect face tracer and face flux directly
    # ------------------------------------------------------------------
    theta_BSO_list = []
    salt_BSO_list = []
    flux_BSO_list   = []
    
    # horizontal faces (u-faces)
    for j, i in zip(y_bsoh, x_bsoh):
        theta_BSO_list.append(theta_x[:, j, i].ravel())
        salt_BSO_list.append(salt_x[:, j, i].ravel())
        flux_BSO_list.append(ADVx[:, j, i].ravel())     # + into basin
    
    # vertical faces (v-faces)
    for j, i in zip(y_bsov, x_bsov):
        theta_BSO_list.append(theta_y[:, j, i].ravel())
        salt_BSO_list.append(salt_y[:, j, i].ravel())
        flux_BSO_list.append((-ADVy[:, j, i]).ravel())  # + into basin
    
    theta_BSO = np.concatenate(theta_BSO_list) if theta_BSO_list else np.array([], dtype=float)
    salt_BSO = np.concatenate(salt_BSO_list) if salt_BSO_list else np.array([], dtype=float)
    flux_BSO   = np.concatenate(flux_BSO_list)   if flux_BSO_list   else np.array([], dtype=float)
    
    ADVh_BSO    = bin_gate_by_face_theta_zero(salt_BSO, flux_BSO, binmidS, nSm1)
    ADVh_BSO_TS = bin_gate_by_face_TS_zero(theta_BSO, salt_BSO, flux_BSO, binmidT, binmidS, nTm1, nSm1)
    G_BSO_TS    = ADVh_BSO_TS / (binwidthT1[:, None] * binwidthS1[None, :])
    
    # ------------------------------------------------------------------
    # FJNZ
    # use the same gate convention as your validated version:
    # ADV_FJNZ[:, y_fjnz, x_fjnzv[0]-1] = -ADVx[:, y_fjnz, x_fjnzv[0]]
    # so bin by tracer at the actual entry face (y_fjnz, x_fjnzv[0])
    # ------------------------------------------------------------------
    theta_FJNZ_list = []
    salt_FJNZ_list = []
    flux_FJNZ_list   = []
    
    for j in y_fjnz:
        theta_FJNZ_list.append(theta_x[:, j, x_fjnzv[0]].ravel())
        salt_FJNZ_list.append(salt_x[:, j, x_fjnzv[0]].ravel())
        flux_FJNZ_list.append((-ADVx[:, j, x_fjnzv[0]]).ravel())
    
    theta_FJNZ = np.concatenate(theta_FJNZ_list) if theta_FJNZ_list else np.array([], dtype=float)
    salt_FJNZ = np.concatenate(salt_FJNZ_list) if salt_FJNZ_list else np.array([], dtype=float)
    flux_FJNZ   = np.concatenate(flux_FJNZ_list)   if flux_FJNZ_list   else np.array([], dtype=float)
    
    ADVh_FJNZ = bin_gate_by_face_theta_zero(salt_FJNZ, flux_FJNZ, binmidS, nSm1)
    ADVh_FJNZ_TS = bin_gate_by_face_TS_zero(theta_FJNZ, salt_FJNZ, flux_FJNZ, binmidT, binmidS, nTm1, nSm1)
    G_FJNZ_TS    = ADVh_FJNZ_TS  / (binwidthT1[:, None] * binwidthS1[None, :])
    
    # ------------------------------------------------------------------
    # SPFJ
    # same sign/index conventions as your validated version
    # ------------------------------------------------------------------
    theta_SPFJ_list = []
    salt_SPFJ_list = []
    flux_SPFJ_list   = []
    
    # ADV_SPFJ[:,y_spfjv,x_spfjv-1] -= ADVx[:, y_spfjv, x_spfjv]
    for j, i in zip(y_spfjv, x_spfjv):
        theta_SPFJ_list.append(theta_x[:, j, i].ravel())
        salt_SPFJ_list.append(salt_x[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVx[:, j, i]).ravel())
    
    # ADV_SPFJ[:,y_spfjh-1,x_spfjh] -= ADVy[:, y_spfjh, x_spfjh]
    for j, i in zip(y_spfjh, x_spfjh):
        theta_SPFJ_list.append(theta_y[:, j, i].ravel())
        salt_SPFJ_list.append(salt_y[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVy[:, j, i]).ravel())
    
    # ADV_SPFJ[:,y_spfjb-1,x_spfjb] -= ADVy[:, y_spfjb, x_spfjb]
    for j, i in zip(y_spfjb, x_spfjb):
        theta_SPFJ_list.append(theta_y[:, j, i].ravel())
        salt_SPFJ_list.append(salt_y[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVy[:, j, i]).ravel())
    
    # ADV_SPFJ[:,y_spfjb,x_spfjb-1] -= ADVx[:, y_spfjb, x_spfjb]
    for j, i in zip(y_spfjb, x_spfjb):
        theta_SPFJ_list.append(theta_x[:, j, i].ravel())
        salt_SPFJ_list.append(salt_x[:, j, i].ravel())
        flux_SPFJ_list.append((-ADVx[:, j, i]).ravel())
    
    theta_SPFJ = np.concatenate(theta_SPFJ_list) if theta_SPFJ_list else np.array([], dtype=float)
    salt_SPFJ = np.concatenate(salt_SPFJ_list) if salt_SPFJ_list else np.array([], dtype=float)
    flux_SPFJ   = np.concatenate(flux_SPFJ_list)   if flux_SPFJ_list   else np.array([], dtype=float)
    
    ADVh_SPFJ = bin_gate_by_face_theta_zero(salt_SPFJ, flux_SPFJ, binmidS, nSm1)
    ADVh_SPFJ_TS = bin_gate_by_face_TS_zero(theta_SPFJ, salt_SPFJ, flux_SPFJ, binmidT, binmidS, nTm1, nSm1)
    G_SPFJ_TS    = ADVh_SPFJ_TS  / (binwidthT1[:, None] * binwidthS1[None, :])
    
    # ------------------------------------------------------------------
    # NZRU
    # validated version:
    # ADV_NZRU[:, j, i-1] -= ADVx[:, j, i]
    # so bin by tracer at actual entry x-face (j, i)
    # ------------------------------------------------------------------
    theta_NZRU_list = []
    salt_NZRU_list = []
    flux_NZRU_list   = []
    
    for j, i in zip(y_nzruv, x_nzruv):
        theta_NZRU_list.append(theta_x[:, j, i].ravel())
        salt_NZRU_list.append(salt_x[:, j, i].ravel())
        flux_NZRU_list.append((-ADVx[:, j, i]).ravel())
    
    theta_NZRU = np.concatenate(theta_NZRU_list) if theta_NZRU_list else np.array([], dtype=float)
    salt_NZRU = np.concatenate(salt_NZRU_list) if salt_NZRU_list else np.array([], dtype=float)
    flux_NZRU   = np.concatenate(flux_NZRU_list)   if flux_NZRU_list   else np.array([], dtype=float)
    
    ADVh_NZRU = bin_gate_by_face_theta_zero(salt_NZRU, flux_NZRU, binmidS, nSm1)
    ADVh_NZRU_TS = bin_gate_by_face_TS_zero(theta_NZRU, salt_NZRU, flux_NZRU, binmidT, binmidS, nTm1, nSm1)
    G_NZRU_TS    = ADVh_NZRU_TS  / (binwidthT1[:, None] * binwidthS1[None, :])
    
    Msum = {}
    Msum["BSO"]  = ADVh_BSO
    Msum["FJNZ"] = ADVh_FJNZ
    Msum["SPFJ"] = ADVh_SPFJ
    Msum["NZRU"] = ADVh_NZRU
    Msum["Msum"] = ADVh_BSO + ADVh_FJNZ + ADVh_SPFJ + ADVh_NZRU
    
    Msum["BSO_TSnn"]  = ADVh_BSO_TS
    Msum["FJNZ_TSnn"] = ADVh_FJNZ_TS
    Msum["SPFJ_TSnn"] = ADVh_SPFJ_TS
    Msum["NZRU_TSnn"] = ADVh_NZRU_TS
    Msum["Msum_TSnn"] = ADVh_BSO_TS + ADVh_FJNZ_TS + ADVh_SPFJ_TS + ADVh_NZRU_TS
    
    Msum["BSO_TS"]  = G_BSO_TS
    Msum["FJNZ_TS"] = G_FJNZ_TS
    Msum["SPFJ_TS"] = G_SPFJ_TS
    Msum["NZRU_TS"] = G_NZRU_TS
    Msum["Msum_TS"] = G_BSO_TS + G_FJNZ_TS + G_SPFJ_TS + G_NZRU_TS

    ####################################################################################################

    ############################################################
    # load the other terms, copy from below

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
    sptop = mk3D_mod(oceSPflx,oceSPtnd) - np.cumsum(oceSPtnd, axis=0)        # we include this in our zconv_top term
    sptop = sptop * RAC3        # g/s
    
    zconv_top_salt = (SFLUX + oceSPflx) * RAC               # g/s
    
    def surface_contrib_JT(zconv_top_salt, sptop, rho, fill_last=0.0):
        """
        zconv_top_heat: (ny, nx)
        swtop:          (nz, ny, nx)
        rcp:            scalar
        fill_last:      value for bottom slice (k = nz-1), usually 0.0 or np.nan
        returns:
          JsurfT:       (nz, ny, nx)  # Sv / PSU
        """
        nz, ny, nx = sptop.shape
    
        eS = zconv_top_salt.reshape(1, ny, nx)  # (1,ny,nx) for broadcast
    
        J = np.empty_like(sptop, dtype=float)
    
        # k = 0: (eT - fT[1]) / rcp / dT / dS * 1e-6
        J[0] = (eS[0] - sptop[1]) / rho if np.ndim(binwidthS)==0 else \
               (eS[0] - sptop[1]) / rho
    
        # 1 .. nz-2: -(fT[k+1]-fT[k]) / rcp / dT / dS * 1e-6
        J[1:nz] = -(sptop[1:nz] - sptop[0:nz-1]) / rho
    
        # bottom slice (k = nz-1): no k+1; choose your boundary convention
        J[-1] = fill_last
        return J
    
    Ft_surftest = surface_contrib_JT(zconv_top_salt,sptop,myparms['rhoconst'])    # this is in PSU.m^3/s
    
    # do the vertical convergence for KPP
    trWtopKPP = -(KPPg_SLT)         # PSU.m^3/s
    
    tmpkpp = np.full((nz,ny,nx),np.nan)
    tmpkpp[:-1,:,:] = trWtopKPP[:-1] - trWtopKPP[1:]
    
    file_name = 'budg3d_snap_set2'
    meta_budg3d_snap_set2 = parsemeta(dirdiags + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_budg3d_snap_set2["fldList"])
    varnames = np.array(["SALTDR"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    
    
    SALTDR = np.full((len(tsstr),nz,ny,nx),np.nan)
    for i in range(len(tsstr)):
        thisSALTDR,its,meta = rdmds(os.path.join(dirdiags, file_name),int(tsstr[i]),returnmeta=True,rec=recs[0])
        thisSALTDR = thisSALTDR.reshape(nz,ny,nx)
        SALTDR[i] = thisSALTDR
    
    SALTDR =  (SALTDR[1, :, :,:] - SALTDR[0, :,:, :]) / dt    # PSU.m/s
    #print(np.nansum(SALTDR),dt)
    
    tmptend = (SALTDR - 0) * mk3D_mod(RAC,SALTDR)    # PSU.m/s * m^2 = PSU.m^3/s

    ############################################################
    # write these to a dF_Snew, so we can output and verify
    # return a 3D dictionary of these terms to verify

    termsS3D = {}
    termsS3D["ADVh"] = ADVhS
    termsS3D["ADVr"] = ADVrS
    termsS3D["DFhS"] = DFhS
    termsS3D["DFrS"] = DFrS
    termsS3D["surf"] = Ft_surftest
    termsS3D["KPP"] = tmpkpp
    termsS3D["tend"] = tmptend

    ############################################################


    # define the ADVh total for this mymsk2
    G_S_offline_new = np.zeros((7, nS-1))
    dF_Snew = np.zeros((7, nS-1))
    Lijnew = np.zeros((7, nS-1), dtype=int)
    
    # also mask these by mymsk3
    # flatten the 3D arrays along all dimensions, as MATLAB’s tmp(:) does
    S_flat    = np.ravel(SALT* mymsk3d, order='F')
    ADVh_flat = np.ravel(ADVhS* mymsk3d,  order='F')
    ADVr_flat = np.ravel(ADVrS* mymsk3d,  order='F')
    DFh_flat = np.ravel(DFhS* mymsk3d,  order='F')
    DFr_flat = np.ravel(DFrS* mymsk3d,  order='F')
    surf_flat = np.ravel(Ft_surftest* mymsk3d,  order='F')
    kpp_flat = np.ravel(tmpkpp* mymsk3d,  order='F')
    tend_flat = np.ravel(tmptend* mymsk3d,  order='F')
    
    # if mapping is True, just return the closure terms from before as tracer.m^3/s
    if mapping:
        return ADVhS,ADVrS,DFhS,DFrS,Ft_surftest,tmpkpp,tmptend
    
    for i in range(nT-1):
        # MATLAB: ij = find(tmp(:) >= bbb.binmidT(i) & tmp(:) < bbb.binmidT(i+1))
        ij = np.where((S_flat >= binmidS[i]) & (S_flat < binmidS[i + 1]))[0]
        Lijnew[0, i] = len(ij)
    
        if len(ij) > 0:
            # MATLAB: dF_Tnew(4,i)=sum(ff.advh(ij)); dF_Tnew(5,i)=sum(ff.advr(ij));
            dF_Snew[0, i] = np.nansum(ADVh_flat[ij])
            dF_Snew[1, i] = np.nansum(ADVr_flat[ij])
            dF_Snew[2, i] = np.nansum(DFh_flat[ij])
            dF_Snew[3, i] = np.nansum(DFr_flat[ij])
            dF_Snew[4, i] = np.nansum(surf_flat[ij])
            dF_Snew[5, i] = np.nansum(kpp_flat[ij])
            dF_Snew[6, i] = np.nansum(tend_flat[ij])
    
    # MATLAB: G_T_offline_new = dF_Tnew ./ repmat(bbb.binwidthT1,[6 1])
    G_S_offline_new = dF_Snew / binwidthS1[None, :]
    
    return Msum,termsS3D,dF_Snew  # these will be in units of m^3/s and PSU.m^3/s
