import numpy as np

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
