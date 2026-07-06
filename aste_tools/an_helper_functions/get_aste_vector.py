
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

    Unew[:,nfy[0]:nfy[0]+nx,nx:2*nx] = -tmpV
    Vnew[:,nfy[0]:nfy[0]+nx,nx:2*nx] = tmpU

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
    Unew = uq    # defined at the western edge
    Vnew = vq    # defined at the southern edge

    return Unew,Vnew
