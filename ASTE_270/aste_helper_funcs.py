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
    a=np.zeros((nz,nfy[0]+nx+nfx[3],2*nx))       #(50,900,270)
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
    #add a new dimension in case it's only 2d field:
    sz=np.shape(fld.f1)
    sz=np.array(sz)
    if(len(sz)<3):
       sz=np.append(sz,1)

    nz=sz[0]
    nx=sz[-1]

    fldo=np.zeros((nz,2*nfy[0]+nx+nfx[3],nx))
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
    # check and fix if 2D
    sz=np.shape(fld)
    print("SZ!",sz)
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
    print("shape of fld:", fld.shape)
    
    nx = nfx[2]
    tmp1 = fld[:,:nfy[0],nx:]
    # tmp2 is Pacific so it's not here
    
    # cw rotation
    tmp3 = fld[:,nfy[0]:nfy[0]+nx,nx:]
    tmp3=np.transpose(tmp3, (1,2,0))
    tmp3 = list(zip(*tmp3))[::-1]
    tmp3 = np.asarray(tmp3)
    tmp3 = np.transpose(tmp3,[2,0,1])
    print(tmp3.shape)
    # plt.pcolormesh(tmp3[0,:,:])
    
    # cw rotation
    tmp4 = fld[:,nfy[0]+nx:,nx:]
    tmp4=np.transpose(tmp4, (1,2,0))
    tmp4 = list(zip(*tmp4))[::-1]
    tmp4 = np.asarray(tmp4)
    tmp4 = np.transpose(tmp4,[2,0,1])
    print(tmp4.shape)
    # plt.pcolormesh(tmp4[0,:,:])
    
    # ccw rotation
    tmp5 = fld[:,0:nfy[0],0:nx]
    tmp5=np.transpose(tmp5, (1,2,0))
    tmp5 = list(zip(*tmp5[::-1]))
    tmp5 = np.asarray(tmp5)
    tmp5 = np.transpose(tmp5,[2,0,1])
    print(tmp5.shape)
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