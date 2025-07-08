import numpy as np
import ctypes
import multiprocessing as mp

def convert_type(nptype):
    '''
    return type, number of bytes
    '''
    if nptype == np.float32:
        return ctypes.c_float, 4
    if nptype == np.float64:
        return ctypes.c_double, 8
    if nptype == np.uint8:
        return ctypes.c_uint8, 1
    return None
 
class RAMBufferBase(object):
    # the buffer to store the sequential data
    def __init__(self, datatype, verbose=False):
        '''
        datatype: np datatype
        datasize: a tuple
        in general, the buffer is in the format of (n x c x h x w) or (n x h x w)
        '''
        self.ctype, self.databyte = convert_type(datatype)
        assert self.ctype is not None, "Type Error {}".format(datatype)

        self.datatype = datatype
        self.datasize = (0)
        # self.reset(datasize)
        self.verbose = verbose
    
    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def reset(self, datasize):
        if datasize != self.datasize: # re-allocate the buffer only if the datasize changes
            # print(datasize, self.datasize, datasize == self.datasize)
            datanum = int(np.prod(datasize))
            self.datasize = datasize
            buffer_base = mp.Array(self.ctype, datanum)
            self.buffer = np.ctypeslib.as_array(buffer_base.get_obj())
            self.buffer = self.buffer.reshape(self.datasize)
            self.vprint("RAM Buffer allocated size {}, mem {} G".format(datasize, datanum * self.databyte / 1000./1000./1000.))

    def insert(self, index, data):
        # assert data.shape == self.datasize[1:], "Insert data shape error! Data shape {}, buffer shape {}".format(data.shape, self.datasize)
        expected = self.datasize[1:]
        if not (len(expected) == 2 and expected[1] == 3 and data.ndim == 2):
            assert data.shape == expected, "Insert data shape error! Data shape {}, buffer shape {}".format(data.shape, self.datasize)


        assert data.dtype == self.datatype, "Insert data type error! Data type {}, buffer type {}".format(data.dtype, self.datatype)
        if data.shape == expected:
            self.buffer[index] = data   
        if data.ndim == 2 and expected[-1] == data.shape[-1]:
            max_pts = expected[0]
            num_pts = data.shape[0]
            if num_pts > max_pts:
                self.buffer[index, :, :] = data[:max_pts, :]
            else:
                self.buffer[index, :num_pts, :] = data




    def load(self, data, startind):
        '''
        load a log of data and pad zero if necessary
        data: numpy array
        '''
        data_framenum = data.shape[0] # the lengh of new data
        endind = data_framenum + startind
        assert  endind <= self.datasize[0], \
            'Error: RAMBuffer load data number {} is bigger than the buffer size {}'.format(data_framenum, self.datasize[0])
        self.buffer[startind:endind] = data

    def __getitem__(self, index):
        # assert index < self.datasize[0], 'Invalid index {}, buffer size {}'.format(index, self.datasize[0])
        return self.buffer[index].copy()

if __name__=="__main__":
    import ipdb;ipdb.set_trace()
    rambuffer = RAMBufferBase(np.float32, (10,3,4,2))
    rambuffer.insert(0, np.random.rand(3,4,2))
    rambuffer.insert(3, np.random.rand(3,4,2))
    rambuffer.load(np.random.rand(8,3,4,2))
    rambuffer.load(np.random.rand(12,3,4,2))
    print(rambuffer.buffer)