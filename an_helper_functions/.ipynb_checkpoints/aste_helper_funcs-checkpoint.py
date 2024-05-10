import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
import numpy.ma as ma
import glob
import os
import re
from operator import mul

class structtype():
    pass

def get_aste_tracer(fldin,nfx,nfy):
    '''
    Inputs:
        fldin: data field in compact coords from mitgcm output (of shape from rdmds reshaped to ny,nx or nz,ny,nx)
        nfx: number of x faces, nfx = np.array([nx, 0 , nx, ncut2 ,ncut1])
        nfy: number of y faces, nfy = np.array([ncut1, 0 , nx, nx, nx])

    outputs:
        the input field reshaped into tracer form, plottable in xyz space

    '''
    
    sz=np.shape(fldin)
    sz=np.array(sz)
    if(len(sz)<3):
       sz=np.append(1,sz)
    
    nz=sz[0]
    nx=sz[-1]

    #print(nz,nx)
    
    #add a new dimension in case it's only 2d field:
    if nz == 1:
        #print('fix 2d')
        fldin=fldin[np.newaxis, :, :]
    #defining a big face:
    a = np.full((nz, nfy[0]+nx+nfx[3], 2*nx), np.nan)
    #print(np.shape(a))
    
    #face1
    tmp=fldin[:,0:nfy[0],0:nx]        #(50,450,270)
    #print(np.shape(tmp))
    # print(np.shape(a[:,0:nfy[0],nx:2*nx]))
    a[:,0:nfy[0],nx:2*nx]=tmp
    # return a
    
    #face3
    tmp=fldin[:,nfy[0]:nfy[0]+nx,0:nx] #(50, 270,270)
    tmp=np.transpose(tmp, (1,2,0))     #(270,270,50)
    ##syntax to rotate cw:
    tmp1=list(zip(*tmp[::-1]))         #type is <class 'zip'> --> <class 'list'>
    tmp1=np.transpose(tmp1,[2,0,1])    #(50,270,270)
    a[:,nfy[0]:nfy[0]+nx,nx:2*nx]=tmp1
    
    #face4
    tmp=np.reshape(fldin[:,nfy[0]+nx:nfy[0]+nx+nfx[3],0:nx],[nz,nx,nfx[3]]) #(50,270,180)
    tmp=np.transpose(tmp, (1,2,0))
    #syntax to rotate cw:
    tmp1=list(zip(*tmp[::-1]))      #type is <class 'list'>
    tmp1=np.asarray(tmp1)           #type <class 'numpy.ndarray'>, shape (180,270,50)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50,180,270)
    a[:,nfy[0]+nx:nfy[0]+nx+nfx[3],nx:2*nx]=tmp1
    
    #face5
    tmp=np.reshape(fldin[:,nfy[0]+nx+nfx[3]:nfy[0]+nx+nfx[3]+nfx[4],0:nx],[nz,nx,nfx[4]]) #(50,270,450)
    tmp=np.transpose(tmp, (1,2,0))
    #syntax to rotate ccw:
    tmp1=list(zip(*tmp))[::-1]      #type is <class 'zip'> --> <class 'list'>
    tmp1=np.asarray(tmp1)           #type <class 'numpy.ndarray'>, shape (450,270,50)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50,450,270)
    a[:,0:nfx[4],0:nx]=tmp1
    
    return a

def aste_faces2compact(fld,nfx,nfy):
    '''
    Reverse of get_aste_faces, taking an input field from tracer form to compact form
    '''
    
    #add a new dimension in case it's only 2d field:
    sz=np.shape(fld.f1)
    sz=np.array(sz)
    if(len(sz)<3):
       sz=np.append(sz,1)

    nz=sz[0]
    nx=sz[-1]

    fldo = np.full((nz, 2 * nfy[0] + nx + nfx[3], nx), np.nan)

    if nz == 1:
        fld.f1=fld.f1[np.newaxis, :, :]
    fldo[:,0:nfy[0],:]=fld.f1
    if nz == 1:
        fld.f3=fld.f3[np.newaxis, :, :]
    fldo[:,nfy[0]:nfy[0]+nfy[2],:]=fld.f3
    if nz == 1:
        fld.f4=fld.f4[np.newaxis, :, :]
    fldo[:,nfy[0]+nfy[2]:nfy[0]+nfy[2]+nfx[3],:]=np.reshape(fld.f4,[nz,nfx[3],nfy[3]])
    if nz == 1:
        fld.f5=fld.f5[np.newaxis, :, :]
    fldo[:,nfy[0]+nfy[2]+nfx[3]:nfy[0]+nfy[2]+nfx[3]+nfx[4],:]=np.reshape(fld.f5,[nz,nfx[4],nfy[4]])

    print(fldo.shape)

    return fldo

def get_aste_faces(fld,nfx,nfy):
    '''
    From big ASTE, get the data on the individual faces from the ASTE grid in case we want to observe individually
    input fld (of shape from rdmds reshaped to ny,nx or nz,ny,nx)
    '''

    
    nx=nfx[0]
    print("nx",nx)
    
    #check the klevel dimension, if 2d, add a third dim
    sz=np.shape(fld)
    sz=np.array(sz)
    print("sz",sz)
    if(len(sz)<3):
        fld=np.copy(fld[np.newaxis,:,:])
    print(fld.shape)
    
    tmp = fld[:,0:nfy[0],0:nx]
    print("tmp",tmp.shape)
    fldout = structtype()
    fldout.f1=fld[:,0:nfy[0],0:nx]                    #face 1
    #fldout = {}
    #fldout["f1"] = fld[:,0:nfy[0],0:nx] 
    fldout.f3=fld[:,nfy[0]:nfy[0]+nfy[2],0:nx]        ##face 3
    fldout.f4=np.reshape(fld[:,nfy[0]+nfy[2]:nfy[0]+nfy[2]+nfx[3],0:nx],[-1,nx,nfx[3]]) ##face 4
    fldout.f5=np.reshape(fld[:,nfy[0]+nfy[2]+nfx[3]:nfy[0]+nfy[2]+nfx[3]+nfx[4],0:nx],[-1,nx,nfx[4]]) ##face 5
    
    return fldout

def plot_aste_faces(fld,nfx,nfy,klev,climit,step):
    '''
    Plots faces 1-4 of the ASTE grid, 
    input
        fld: must be from rdmds do not edit or reshape this
    '''
    fldout=get_aste_faces(fld,nfx,nfy)
    nx=nfx[0]
    #step=(climit[1]-climit[0])/100
    print(step)
    clevels = np.arange(climit[0], climit[1], step)
    fig,axs=plt.subplots(2,2)
    pcm=axs[0,0].contourf(fldout.f1[klev-1,:,:],levels=clevels, cmap='viridis')
    fig.colorbar(pcm,ax=axs[0,0],location='right')
    axs[0,0].title.set_text('fld face1')
    pcm=axs[0,1].contourf(fldout.f3[klev-1,:,:],levels=clevels,cmap='viridis')
    fig.colorbar(pcm,ax=axs[0,1],location='right')
    axs[0,1].title.set_text('fld face3')
    pcm=axs[1,0].contourf(fldout.f4[klev-1,:,:],levels=clevels,cmap='viridis')
    fig.colorbar(pcm,ax=axs[1,0],location='right')
    axs[1,0].title.set_text('fld face4')
    pcm=axs[1,1].contourf(fldout.f5[klev-1,:,:],levels=clevels,cmap='viridis')
    fig.colorbar(pcm,ax=axs[1,1],location='right')
    axs[1,1].title.set_text('fld face5')

def aste_tracer2compact(fld, nfx, nfy):
    '''
    Reverse of get_aste_tracer function
    Inputs:
        fld: the field in tracer form [nx*2 nfy(1)+nfy(3)+nfx(4)+nfx(5),nz]
        nfx: number of x faces
        nfy: number of y faces

    Outputs:
        fldout: the original data field in compact form, useful for comparison with read binary files
        Out: compact format [nz 1350 270]
    '''
    # check and fix if 2D
    sz=np.shape(fld)
    #print("SZ!",sz)
    sz=np.array(sz)
    #if(len(sz)<3):
    #   sz=np.append(1,sz)

    #add a new dimension in case it's only 2d field:
    if(len(sz)<3):
        sz=np.append(1,sz)
        print('fix 2d')
        fld=fld[np.newaxis, :, :]
        
    nz=sz[0]
    nx=sz[-1]
    print("shape of tracer fld:", fld.shape)
    
    nx = nfx[2]
    tmp1 = fld[:,:nfy[0],nx:]
    # tmp2 is Pacific so it's not here
    
    # cw rotation
    tmp3 = fld[:,nfy[0]:nfy[0]+nx,nx:]
    tmp3=np.transpose(tmp3, (1,2,0))
    tmp3 = list(zip(*tmp3))[::-1]
    tmp3 = np.asarray(tmp3)
    tmp3 = np.transpose(tmp3,[2,0,1])
    #print(tmp3.shape)
    # plt.pcolormesh(tmp3[0,:,:])
    
    # cw rotation
    tmp4 = fld[:,nfy[0]+nx:,nx:]
    tmp4=np.transpose(tmp4, (1,2,0))
    tmp4 = list(zip(*tmp4))[::-1]
    tmp4 = np.asarray(tmp4)
    tmp4 = np.transpose(tmp4,[2,0,1])
    #print(tmp4.shape)
    # plt.pcolormesh(tmp4[0,:,:])
    
    # ccw rotation
    tmp5 = fld[:,0:nfy[0],0:nx]
    tmp5=np.transpose(tmp5, (1,2,0))
    tmp5 = list(zip(*tmp5[::-1]))
    tmp5 = np.asarray(tmp5)
    tmp5 = np.transpose(tmp5,[2,0,1])
    #print(tmp5.shape)
    # plt.pcolormesh(tmp5[0,:,:])
    # we now have 5 separate faces

    tmp_struct = structtype()
    tmp_struct.f1 = tmp1
    tmp_struct.f3 = tmp3
    tmp_struct.f4 = tmp4
    tmp_struct.f5 = tmp5
    
    compact = aste_faces2compact(tmp_struct,nfx,nfy)
    print("compact shape",compact.shape)  # this is rdmds shape

    return compact

def get_aste_vector(U,V,nfx,nfy,sign_switch):
    '''
    inputs:
        U: in compact form
        V: in compact form
        nfx: x-size of ASTE faces
        nfy: y-size of ASTE faces
        sign_switch: boolean, TF
    outputs:
        uaste, vaste: the tracer form of the u and v arrays of size 541,901
    '''
    # take care of 2D if needed
    if len(U.shape)<3:
        U = U[np.newaxis,:,:]
    if len(V.shape)<3:
        V = V[np.newaxis,:,:]
    
    # set up the size
    nz,ny,nx = U.shape[0],U.shape[1],U.shape[2]   #print(nz,ny,nx)
    #print(nz,ny,nx)
    
    # order stored in compact format
    nfx1 = nfx.copy()
    nfy1 = nfy.copy()

    # try to put all on one big matrix - we skip edge cases for now
    Unew = np.full((nz,nfy[0] + nfx[2] + nfx[3],nfy[4] + nfx[0]),np.nan)           # nz, 900, 540
    Vnew = np.full((nz, nfy[0] + nfx[2] + nfx[3],nfy[4] + nfx[0]), np.nan)

    # face1
    tmpU=U[:,0:nfy[0],0:nx]               #(nz,450,270)
    tmpV=V[:,0:nfy[0],0:nx]               #(nz,450,270)
    Unew[:,0:nfy[0],nx:2*nx]=tmpU
    Vnew[:,0:nfy[0],nx:2*nx]=tmpV

    # face 3 - rotate 180 degrees
    # -u -> new_u, -v -> new_v
    tmpU=U[:,nfy[0]:nfy[0]+nx,0:nx]         #(nz, 270,270)
    tmpU=np.transpose(tmpU, (1,2,0))        #(270,270,nz)
    tmpU=list(zip(*tmpU[::-1]))
    tmpU=np.transpose(tmpU,[2,0,1])         #(nz,270,270)

    tmpV=V[:,nfy[0]:nfy[0]+nx,0:nx]         #(nz, 270,270)
    tmpV=np.transpose(tmpV, (1,2,0))        #(270,270,nz)
    tmpV=list(zip(*tmpV[::-1]))
    tmpV=np.transpose(tmpV,[2,0,1])         #(nz,270,270)

    Unew[:,nfy[0]:nfy[0]+nx,nx:2*nx] = -tmpU
    Vnew[:,nfy[0]:nfy[0]+nx,nx:2*nx] = -tmpV
    
    # WE DIDN'T DO FACE 3 ROTATED 90DEG CCW - fix this
    # u - new_v, -v -> new_u
    #Unew(nfy(5)+1:nfy(5)+nfx(1),nfy(1)+1:nfy(1)+nfx(3),:)=-sym_g_mod(ffv{3},5,0);    # from MATLAB
    #Vnew(nfy(5)+1:nfy(5)+nfx(1),nfy(1)+1:nfy(1)+nfx(3),:)= sym_g_mod(ffu{3},5,0);    # from MATLAB
    #print(U.shape,V.shape)
    #tmpU = U[:,nfy[0]:nfy[0]+nfx[2],nfy[4]:(nfy[4]+nfx[0])]
    #tmpV = V[:,nfy[0]:nfy[0]+nfx[2],nfy[4]:(nfy[4]+nfx[0])]
    #print(tmpU.shape,tmpV.shape)

    # face 4 - rot 90 degrees ccw
    # u -> new_v, -v -> new_u
    tmpU=np.reshape(U[:,nfy[0]+nx:nfy[0]+nx+nfx[3],0:nx],[nz,nx,nfx[3]])    #(nz,270,180)
    tmpU=np.transpose(tmpU, (1,2,0))
    tmpU=list(zip(*tmpU[::-1]))
    tmpU=np.asarray(tmpU)
    tmpU=np.transpose(tmpU,[2,0,1])                                         #(nz,180,270)

    tmpV=np.reshape(V[:,nfy[0]+nx:nfy[0]+nx+nfx[3],0:nx],[nz,nx,nfx[3]])    #(nz,270,180)
    tmpV=np.transpose(tmpV, (1,2,0))
    tmpV=list(zip(*tmpV[::-1]))
    tmpV=np.asarray(tmpV)
    tmpV=np.transpose(tmpV,[2,0,1])                                         #(nz,180,270)

    Unew[:,nfy[0]+nx:nfy[0]+nx+nfx[3],nx:2*nx]=-tmpV
    Vnew[:,nfy[0]+nx:nfy[0]+nx+nfx[3],nx:2*nx]= tmpU

    # face 5
    # v -> new_u, -u -> new_v
    tmpU=np.reshape(U[:,nfy[0]+nx+nfx[3]:nfy[0]+nx+nfx[3]+nfx[4],0:nx],[nz,nx,nfx[4]]) #(nz,270,450)
    tmpU=np.transpose(tmpU, (1,2,0))
    tmpU=list(zip(*tmpU))[::-1]
    tmpU=np.asarray(tmpU)
    tmpU=np.transpose(tmpU,[2,0,1])                                         #(nz,450,270)

    tmpV=np.reshape(V[:,nfy[0]+nx+nfx[3]:nfy[0]+nx+nfx[3]+nfx[4],0:nx],[nz,nx,nfx[4]]) #(nz,270,450)
    tmpV=np.transpose(tmpV, (1,2,0))
    tmpV=list(zip(*tmpV))[::-1]
    tmpV=np.asarray(tmpV)
    tmpV=np.transpose(tmpV,[2,0,1])                                         #(nz,450,270)

    Unew[:,0:nfx[4],0:nx]=tmpV
    Vnew[:,0:nfx[4],0:nx]=-tmpU

    # swtich sign if needed
    if sign_switch:
        Unew = np.abs(Unew)
        Vnew = np.abs(Vnew)

    # we still need to shift to add the padding in U
    up = Unew.copy()                                                # (nz,900,540)
    sz = up.shape
    #print(sz)
    new_shape = (sz[0], sz[1]+1, sz[2] + 1)
    uq = np.full(new_shape, np.nan)                                 # (nz,901,541)
    
    # keep everything in y-dir from 1-450 (top of face 3), ix goes from 1-540, keep 541 as nan
    uq[:,:nfy[0],:nx*2] = up[:,:nfy[0],:nx*2]
    #for y-dir 450, what was called ix=1 should now be reassigned ix=2, shift everything 1 grid to the right
    uq[:,nfy[0]:sz[1],1:nx*2+1] = up[:,nfy[0]:sz[1],:nx*2]

    # now we can add the padding in V
    vp = Vnew.copy()
    vq = np.full(new_shape, np.nan)                                 # (nz,901,541)

    # keep everything from ix=271:540, ignore 901
    vq[:,:nfy[0]+nfx[2]+nfx[3],nx:2*nx] = vp[:,:nfy[0]+nfx[2]+nfx[3],nx:2*nx]
    # shift everything from ix=1:270 1 grid up
    vq[:,1:sz[1]+1,:nx] = vp[:,:sz[1],:nx]

    # reset original arrays after padding
    Unew = uq
    Vnew = vq

    return Unew,Vnew

#create a function
def read_aste_float32(filename,nx,ny,nnz):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>f'))
        if nnz > 1:
            fld = np.reshape(data,[nnz, ny, nx])   #elif:
        else:
            fld = np.reshape(data,[ny, nx])
    return fld

def read_aste_float64(filename,nx,ny,nnz):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>f8'))
        if nnz > 1:
            fld = np.reshape(data,[nnz, ny, nx])   #elif:
        else:
            fld = np.reshape(data,[ny, nx])
    return fld

def write_float32(fout,fld):
    #del bytes
    #fld=np.float32(fld)
    with open(fout, 'wb') as file:
        bytes = fld.tobytes()
        file.write(bytes)
        
#create a function
def read_float32(fileIn):
    with open(fileIn, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>f'))
        print(np.shape(data))
    return data