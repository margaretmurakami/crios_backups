import numpy as np
import sys
sys.path.append("/home/mmurakami/MITgcm/MITgcm_c68r/MITgcm-checkpoint68r/utils/python/MITgcmutils/MITgcmutils/")
from mds import *

def read_float32(fileIn):
    with open(fileIn, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>f'))
        #print(np.shape(data))
    return data

def read_float64(fileIn):
    with open(fileIn, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>f8'))
        #print(np.shape(data))
    return data

def read_float32_skip(fileIn,recordLen,recordNo):
    memArray = np.zeros(recordLen, dtype=np.dtype('>f')) # a buffer for 1 record
    with open(fileIn, 'rb') as file:
        # Reading a record recordNo from file into the fldout
        file.seek(recordLen * 4 * recordNo)
        bytes = file.read(recordLen*4)
        fldout = np.frombuffer(bytes, dtype=np.dtype('>f')).copy()
    return fldout

#Now try to create a read function that allows skip:
def read_float64_skip(fileIn,recordLen,recordNo):
    memArray = np.zeros(recordLen, dtype=np.dtype('>f8')) # a buffer for 1 record
    with open(fileIn, 'rb') as file:
        # Reading a record recordNo from file into the fldout
        file.seek(recordLen * 8 * recordNo)
        bytes = file.read(recordLen*8)
        fldout = np.frombuffer(bytes, dtype=np.dtype('>f8')).copy()
    return fldout

# write abbreviation functions
def read_field3d(tsstr,nz,ny,nx,file_name,varnames,mymsk,dirIn):
    FIELD = np.full((len(tsstr),nz,ny,nx),np.nan)
    
    meta_set = parsemeta(dirIn + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_set["fldList"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
        
    for i in range(len(tsstr)):
        read = [int(tsstr[i])]
        FIELDi, its, meta = rdmds(dirIn + '/' + file_name, read, returnmeta=True, rec=recs[0])
        FIELD[i, :, :, :] = np.reshape(FIELDi, (nz, ny, nx)) * mymsk[np.newaxis, :, :]

    return FIELD

# write abbreviation functions
def read_field2d(tsstr,ny,nx,file_name,varnames,mymsk,dirIn):
    FIELD = np.full((len(tsstr),ny,nx),np.nan)
    
    meta_set = parsemeta(dirIn + file_name + "." + tsstr[0] + ".meta")
    fldlist = np.array(meta_set["fldList"])
    recs = np.array([])
    for var in varnames:
        irec = np.where(fldlist == var)
        recs = np.append(recs, irec[0][0])
        
    for i in range(len(tsstr)):
        read = [int(tsstr[i])]
    
        FIELDi,its,meta = rdmds(dirIn + '/' + file_name,read,returnmeta=True,rec=recs[0])
        FIELD[i,:,:] = np.reshape(FIELDi,(ny,nx)) * mymsk[:,:]
    return FIELD
