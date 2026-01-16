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
    # this is not correct because we want to divide by Face T
    G_BSO = ADVh_BSO #/ binwidthT1
    G_FJNZ = ADVh_FJNZ #/ binwidthT1
    G_SPFJ = ADVh_SPFJ #/ binwidthT1
    G_NZRU = ADVh_NZRU #/ binwidthT1

    return G_BSO,G_FJNZ,G_SPFJ,G_NZRU

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

# define the following as a function to use for any ADV and tracer value
def gate_transport(ADVx_TH,ADVy_TH,THETA,nz,ny,nx,y_bsoh,x_bsoh,y_bsov,x_bsov,y_fjnzv,x_fjnzv,y_nzruv,x_nzruv,y_spfjh,x_spfjh,y_spfjv,x_spfjv,y_spfjb,x_spfjb):
    # get the gateway transports for the Barents Sea from ADVx and y of TH or SALT, with the tracer T or S
    
    # --- reshape to 3D ---
    ADVx_TH = ADVx_TH.reshape((nz, ny, nx))   # advective heat flux on x-faces
    ADVy_TH = ADVy_TH.reshape((nz, ny, nx))   # advective heat flux on y-faces
    THETA   = THETA.reshape((nz, ny, nx))     # cell-centered temperature
    
    ADVx_vol = ADVx_TH
    ADVy_vol = ADVy_TH
    
    # ---- BSO ----
    ADV_west = np.zeros((nz, ny, nx))
    
    # horizontal faces (u-faces)
    for j, i in zip(y_bsoh, x_bsoh):
        # flux through x-face at (j,i) mapped into cell (j,i)
        ADV_west[:, j, i] += ADVx_vol[:, j, i]    # + into basin
    
    # vertical faces (v-faces)
    for j, i in zip(y_bsov, x_bsov):
        # flux through y-face at (j,i) mapped into cell (j-1,i)
        ADV_west[:, j-1, i] -= ADVy_vol[:, j, i]  # sign chosen so + into basin
    
    # ---- FJNZ ----
    ADV_FJNZ = np.zeros((nz, ny, nx))
    ADV_FJNZ[:,y_fjnzv,x_fjnzv[0]-1] = -ADVx_vol[:, y_fjnzv, x_fjnzv[0]]
    
    # ---- SPFJ (NZ exit) ----
    ADV_SPFJ = np.zeros((nz, ny, nx))
    ADV_SPFJ[:,y_spfjv,x_spfjv-1] -= ADVx_vol[:, y_spfjv, x_spfjv]
    ADV_SPFJ[:,y_spfjh-1,x_spfjh] -= ADVy_vol[:, y_spfjh, x_spfjh]
    ADV_SPFJ[:,y_spfjb-1,x_spfjb] -= ADVy_vol[:, y_spfjb, x_spfjb]
    ADV_SPFJ[:,y_spfjb,x_spfjb-1] -= ADVx_vol[:, y_spfjb, x_spfjb]  # this fixed the issue
    
    # ---- NZRU (small Russia gate) ----
    ADV_NZRU = np.zeros((nz, ny, nx))
    
    for j, i in zip(y_nzruv, x_nzruv):
        ADV_NZRU[:, j, i-1] -= ADVx_vol[:, j, i]   # + into basin

    return ADV_west,ADV_FJNZ,ADV_SPFJ,ADV_NZRU  # in m^3.tracer/s


def create_layersTHETA(tsstr,mygrid,myparms,dirdiags,dirState,layers_path,mymsk,nz,ny,nx,nfx,nfy):
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
    
    return G_T_offline_new,dF_Tnew  # these will be in units of m^3/s and degC.m^3/s

def create_layersSALT(tsstr,mygrid,myparms,dirdiags,dirState,layers_path,mymsk,nz,ny,nx,nfx,nfy):
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

    ### do for the diffusive and surface terms
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
    
    SALTDR =  (SALTDR[1, :, :,:] - SALTDR[0, :,:, :]) / 1    # PSU.m/s
    #print(np.nansum(SALTDR),dt)
    
    tmptend = (SALTDR - 0) * mk3D_mod(RAC,SALTDR)    # PSU.m/s * m^2 = PSU.m^3/s

    ############################################################
    # write these to a dF_Snew, so we can output and verify
    # redefine all the terms as a list from how we did before

    # define the ADVh total for this mymsk2
    G_S_offline_new = np.zeros((7, nS-1))
    dF_Snew = np.zeros((7, nS-1))
    Lijnew = np.zeros((7, nS-1), dtype=int)
    
    # also mask these by mymsk3
    # flatten the 3D arrays along all dimensions, as MATLAB’s tmp(:) does
    S_flat    = np.ravel(SALT*hf* mymsk3d, order='F')
    ADVh_flat = np.ravel(ADVhS*hf* mymsk3d,  order='F')
    ADVr_flat = np.ravel(ADVrS*hf* mymsk3d,  order='F')
    DFh_flat = np.ravel(DFhS*hf* mymsk3d,  order='F')
    DFr_flat = np.ravel(DFrS*hf* mymsk3d,  order='F')
    surf_flat = np.ravel(Ft_surftest*hf* mymsk3d,  order='F')
    kpp_flat = np.ravel(tmpkpp*hf* mymsk3d,  order='F')
    tend_flat = np.ravel(tmptend*hf* mymsk3d,  order='F')
    
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
    
    return G_S_offline_new,dF_Snew  # these will be in units of m^3/s and degC.m^3/s


def create_gates(tsstr,mygrid,myparms,dirdiags,dirState,layers_path,mymsk,nz,ny,nx,nfx,nfy,y_bsoh,x_bsoh,y_bsov,x_bsov,y_fjnzv,x_fjnzv,y_nzruv,x_nzruv,y_spfjh,x_spfjh,y_spfjv,x_spfjv,y_spfjb,x_spfjb):
    # create the M term the ADV in SLT and TMP for big ASTE
    t2 = int(tsstr[0])

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
    ADVx_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[2])
    ADVy_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[3])

    # we need to also create the theta and salt bins for this area
    # 'diags/state_3d_set1'
    # read theta and salt averages from the t2 timestep (average)
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

    # also load the bins again so we have them locally
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

    ADV_westT,ADV_FJNZT,ADV_SPFJT,ADV_NZRUT = gate_transport(ADVx_TH,ADVy_TH,THETA,nz,ny,nx,y_bsoh,x_bsoh,y_bsov,x_bsov,y_fjnzv,x_fjnzv,y_nzruv,x_nzruv,y_spfjh,x_spfjh,y_spfjv,x_spfjv,y_spfjb,x_spfjb)  # this is the nz,ny,nx with only the gates + as in
    ADV_westS,ADV_FJNZS,ADV_SPFJS,ADV_NZRUS = gate_transport(ADVx_SLT,ADVy_SLT,SALT,nz,ny,nx,y_bsoh,x_bsoh,y_bsov,x_bsov,y_fjnzv,x_fjnzv,y_nzruv,x_nzruv,y_spfjh,x_spfjh,y_spfjv,x_spfjv,y_spfjb,x_spfjb)

    ##########################################################################################
    theta_flat = THETA.ravel()
    salt_flat = SALT.ravel()
    
    G_BSOT,G_FJNZT,G_SPFJT,G_NZRUT = gateway3D(ADV_westT,ADV_FJNZT,ADV_SPFJT,ADV_NZRUT,THETA,binmidT,nTm1)  # in m^3/s
    G_BSOS,G_FJNZS,G_SPFJS,G_NZRUS = gateway3D(ADV_westS,ADV_FJNZS,ADV_SPFJS,ADV_NZRUS,SALT,binmidS,nSm1)

    MsumS = G_BSOS + G_FJNZS + G_SPFJS+ G_NZRUS   # overall transport in S
    MsumT = G_BSOT + G_FJNZT + G_SPFJT + G_NZRUT  # overall transport in T
    
    MsumS[np.isnan(MsumS)] = 0
    MsumT[np.isnan(MsumT)] = 0

    return(G_BSOT,G_FJNZT,G_SPFJT,G_NZRUT, G_BSOS,G_FJNZS,G_SPFJS,G_NZRUS)

# also create a function which takes the path, iwet, tsstr value and returns the G term
def get_G_terms(layers_path,dirgridw,iwet_mine,tsstr,name,iwetC2d,LwetC2d,LwetC,hf1,nz,ny,nx):
    
    
    # add the TS bins
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
    
    # add the grid
    # 2d
    rac2d = read_float64(dirgridw + "RAC_" + str(LwetC2d) + ".data")
    print("RAC2d",rac2d.shape)
    drf3d = read_float32(dirgridw + "DRF_" + str(LwetC) + ".data")
    hf1flat = np.reshape(hf1,hf1.flatten().shape[0])
    hf2d = hf1flat[iwetC2d]
    rac2dtile = np.tile(rac2d,(nT-1,1)) #.shape
    hf2dtile = np.tile(hf2d,(nT-1,1))
    ffac = 1e-6
    
    # we can load the layers diagnostics output here
    ffac=1e-6
    G_T=np.array([])
    metaT=parsemeta(layers_path + "diags/LAYERS/layers_3d_Ttend." + str(tsstr[-1])+ ".meta")
    metaS=parsemeta(layers_path + "diags/LAYERS/layers_3d_Stend." + str(tsstr[-1])+ ".meta")
    nFldsS = metaS["fldList"]
    nFldsT = metaT["fldList"]
    print(len(nFldsS))
    print(len(nFldsT))
    setTtend=nFldsT.copy()
    setStend=nFldsS.copy()
    
    # check for the correct diagnostics
    metaS['fldList']
    a = metaS['fldList']
    # print(a[12],a[13],a[14],a[19],a[20],a[23])  # surf 
    ifldS = np.array([12,13,14,19,20,23])  # total tend, surf, hDiff, vDiff, hADV, vADV
    
    metaT['fldList']   # 13, 14, 15, 16, 19, 22
    b = metaT['fldList']
    # print(b[0],b[1],b[2],b[7],b[8],b[11])
    ifldT = np.array([0,1,2,7,8,11])    # surf, hDiff, vDiff, hADV, vADV, total tend

    # get G_T from layers
    # make the G_T term
    G_T = {}
    G_T[name] = {}
        
    # now loop through
    for ts in tsstr:
        G_T[name][ts] = {}
        for i in range(len(ifldT)-1, -1, -1):
            tmp = read_float32_skip(layers_path + "diags/LAYERS/layers_3d_Ttend." + tsstr[-1] + ".data", nx*ny*(nT-1),ifldT[i])
            tmp = np.reshape(tmp,(nT-1,nx*ny))
            tmp = tmp[:,iwetC2d] * (rac2dtile * hf2dtile) * ffac
            if i == ifldT.shape[0]-1:
                # if LTto2TH, do not remove from residual
                residT = tmp
            else:
                # else if vADV, hADV, vDiff, hDiff, surface, remove from residual
                residT = residT-tmp
        
            # just do the Barents Sea for this one
            a = np.nansum(tmp[:,iwet_mine],axis=1)
            G_T[name][ts][setTtend[ifldT[i]]] = a
    
        G_T[name][ts]["residT"] = np.nansum(residT[:,iwet_mine],axis=1)

    # create G_S for all basins (line 258)
    G_S = {}
    G_S[name] = {}
    
    # now loop through similar to ifldS
    for ts in tsstr:
        G_S[name][ts] = {}
        for i in range(len(ifldS)-1, -1, -1):
            tmp = read_float32_skip(layers_path + "diags/LAYERS/layers_3d_Stend." + tsstr[-1] + ".data", nx*ny*(nT-1),ifldS[i])
            tmp = np.reshape(tmp,(nS-1,nx*ny))
            tmp = tmp[:,iwetC2d] * (rac2dtile * hf2dtile) * ffac
            if i == ifldS.shape[0]-1:
                # if LSto1SLT, do not remove from residual, this is the total tendency
                residS = tmp
            else:
                # else if vADV, hADV, vDiff, hDiff, surface, remove from residual
                residS = residS-tmp
        
            # just do for the Barents Sea
            a = np.nansum(tmp[:,iwet_mine],axis=1)
            G_S[name][ts][setStend[ifldS[i]]] = a
        
        # loop through again to calculate resid
        G_S[name][ts]["residS"] = np.nansum(residS[:,iwet_mine],axis=1)
    
    return G_T,G_S