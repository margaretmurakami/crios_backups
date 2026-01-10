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


def create_layers(tsstr,mygrid,dirdiags,dirState,layers_path,mymsk,nz,ny,nx,nfx,nfy):
    # we want to create dF_Tnew, basically, which contains the information from the layers output mimicked by ASTER1
    # let's just check with ADVh first
    mymsk3d = np.tile(mymsk[np.newaxis,:,:],(nz,1,1))
    t2 = int(tsstr[0])
    hf = mygrid['hFacC']
    
    # load THETA
    file_name = "state_3d_set1"
    meta_state_3d_set1 = parsemeta(layers_path + "diags/STATE/" + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_state_3d_set1["fldList"])
    varnames = np.array(["THETA","SALT"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
    THETA,its,meta = rdmds(os.path.join(layers_path + "diags/STATE/", file_name),t2,returnmeta=True,rec=recs[0])
    SALT,its,meta = rdmds(os.path.join(layers_path + "diags/STATE/", file_name),t2,returnmeta=True,rec=recs[1])
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
    ADVx_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[2])
    ADVy_SLT,its,meta = rdmds(os.path.join(dirdiags, file_name),t2,returnmeta=True,rec=recs[3])
    
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
    ADVr_SLT = ADVr_SLT.reshape(nz,ny,nx)

    # for temp, get convergence
    ADV_hconvT = calc_UV_conv_mod(nfx, nfy,get_aste_faces(ADVx_TH.reshape(nz, ny, nx), nfx, nfy),get_aste_faces(ADVy_TH.reshape(nz, ny, nx), nfx, nfy))
    ADV_hconvT = ADV_hconvT * hf   # degC·m^3/s at cell centers (matches: ff.DFh = ff.DFh .* hf)
    ADVhT = ADV_hconvT
    trWtopADV = -(ADVr_TH)
    ADVrT = np.zeros((nz,ny,nx),dtype=float)
    ADVrT[:-1,:,:] = (trWtopADV[:-1] - trWtopADV[1:])
    # for salt
    ADV_hconvS = calc_UV_conv_mod(nfx, nfy,get_aste_faces(ADVx_SLT.reshape(nz, ny, nx), nfx, nfy),get_aste_faces(ADVy_SLT.reshape(nz, ny, nx), nfx, nfy))
    ADV_hconvS = ADV_hconvS * hf   # degC·m^3/s at cell centers (matches: ff.DFh = ff.DFh .* hf)
    ADVhS = ADV_hconvS
    trWtopADV = -(ADVr_SLT)
    ADVrS = np.zeros((nz,ny,nx),dtype=float)
    ADVrS[:-1,:,:] = (trWtopADV[:-1] - trWtopADV[1:])

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

    # create the G_T term
    # define the ADVh total for this mymsk2
    G_T_offline_new = np.zeros((2, nT-1))
    ADV_Tnew = np.zeros((2, nT-1))
    Lijnew = np.zeros((2, nT-1), dtype=int)
    # also mask these by mymsk3
    # flatten the 3D arrays along all dimensions, as MATLAB’s tmp(:) does
    T_flat    = np.ravel(THETA* mymsk3d, order='F')
    ADVh_flat = np.ravel(ADVhT* mymsk3d,  order='F')
    ADVr_flat = np.ravel(ADVrT* mymsk3d,  order='F')
    for i in range(nT-1):
        # MATLAB: ij = find(tmp(:) >= bbb.binmidT(i) & tmp(:) < bbb.binmidT(i+1))
        ij = np.where((T_flat >= binmidT[i]) & (T_flat < binmidT[i + 1]))[0]
        Lijnew[0, i] = len(ij)
    
        if len(ij) > 0:
            # MATLAB: dF_Tnew(4,i)=sum(ff.advh(ij)); dF_Tnew(5,i)=sum(ff.advr(ij));
            ADV_Tnew[0, i] = np.nansum(ADVh_flat[ij])
            ADV_Tnew[1, i] = np.nansum(ADVr_flat[ij])

    G_T_offline_new = ADV_Tnew / binwidthT1[None, :]

    G_S_offline_new = np.zeros((2, nS-1))
    ADV_Snew = np.zeros((2, nS-1))
    Lijnew = np.zeros((2, nS-1), dtype=int)
    # also mask these by mymsk3
    # flatten the 3D arrays along all dimensions, as MATLAB’s tmp(:) does
    S_flat    = np.ravel(SALT* mymsk3d, order='F')
    ADVh_flat = np.ravel(ADVhS* mymsk3d,  order='F')
    ADVr_flat = np.ravel(ADVrS* mymsk3d,  order='F')
    for i in range(nS-1):
        # MATLAB: ij = find(tmp(:) >= bbb.binmidT(i) & tmp(:) < bbb.binmidT(i+1))
        ij = np.where((S_flat >= binmidS[i]) & (S_flat < binmidS[i + 1]))[0]
        Lijnew[0, i] = len(ij)
    
        if len(ij) > 0:
            # MATLAB: dF_Tnew(4,i)=sum(ff.advh(ij)); dF_Tnew(5,i)=sum(ff.advr(ij));
            ADV_Snew[0, i] = np.nansum(ADVh_flat[ij])
            ADV_Snew[1, i] = np.nansum(ADVr_flat[ij])
    
    # MATLAB: G_T_offline_new = dF_Tnew ./ repmat(bbb.binwidthT1,[6 1])
    G_S_offline_new = ADV_Snew / binwidthS1[None, :]

    return G_T_offline_new,ADV_Tnew,G_S_offline_new,ADV_Snew  # these will be in units of m^3/s and degC.m^3/s