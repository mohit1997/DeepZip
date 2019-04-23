import numpy as np
import struct
import padasip as pa
import os
import subprocess
import argparse
import numpy as np

class NLMS_predictor:        
    def init(self, n):
        self.filt = pa.filters.FilterNLMS(n, mu=1.,w="zeros")
        self.n = n
        return
    def predict(self,past,idx):
        if(idx==0):
            return 0
        elif(idx<=self.n):
            return past[idx-1]
        else:
            self.filt.adapt(past[idx-1],past[idx-self.n-1:idx-1])
            return self.filt.predict(past[idx-self.n:idx])

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-mode', action='store', dest='mode',
                    help='c or d (compress/decompress)', required=True)
parser.add_argument('-infile', action='store', dest='infile', help = 'infile .npy/.7z', type = str, required=True)
parser.add_argument('-outfile', action='store', dest='outfile', help = 'outfile .npy/.7z', type = str, required=True)
parser.add_argument('-n', action='store', dest='n', help = 'order of NLMS filter for compression (default 4)', type = int, default = 4)
parser.add_argument('-maxerror', action='store', dest='maxerror', help = 'max allowed error for compression', type=float)

args = parser.parse_args()

if args.mode == 'c':
    if args.maxerror == None:
        raise RuntimeError('maxerror not specified for mode c')
    tmpfile = args.outfile+'.tmp'
    reconfile = args.outfile+'.recon.npy'
    maxerror = np.float32(args.maxerror)
    # read file
    data = np.load(args.infile)
    data = np.array(data,dtype=np.uint32)
    # initialize quantization (with roughly 65535 bins at the start)
    maxlevel = np.float32(65533*maxerror)
    minlevel = np.float32(-65533*maxerror)
    numbins = int((maxlevel-minlevel)/(2*maxerror))+2
    bins = np.linspace(minlevel,maxlevel,numbins,dtype=np.float32)
    fmtstring = 'H' # 16 bit unsigned
    bin_idx_len = 2 # in bytes

    # initialize predictor
    predictor = NLMS_predictor()
    predictor.init(args.n)
    reconstruction = np.zeros(np.shape(data),dtype=np.uint32)
    f_out = open(tmpfile,'wb')
    # write max error to file (needed during decompression)
    f_out.write(struct.pack('f',maxerror))
    # write length of array to file
    f_out.write(struct.pack('I',len(data)))
    # write n to file
    f_out.write(struct.pack('I',args.n))
    for i in range(len(data)):
        predval = np.float32(predictor.predict(reconstruction,i))
        diff = np.float32(data[i] - predval)
        if (diff > maxlevel + maxerror or diff < minlevel - maxerror):
            f_out.write(struct.pack(fmtstring,numbins))
            f_out.write(struct.pack('f',data[i]))
            reconstruction[i] = data[i]
        else:
            if diff > maxlevel:
                bin_idx = numbins - 1
            elif diff < minlevel:
                bin_idx = 0
            else:
                bin_idx = np.digitize(diff,bins)
                if(bin_idx != numbins and bin_idx != 0):
                    if(np.abs(diff-bins[bin_idx])>np.abs(diff-bins[bin_idx-1])):
                        bin_idx -= 1
                f_out.write(struct.pack(fmtstring,bin_idx))
                reconstruction[i] = predval + bins[bin_idx]
    f_out.close()
    subprocess.run(['7z','a',args.outfile,tmpfile])
    os.remove(tmpfile)
    # save reconstruction to a file (for comparing later)
    np.save(reconfile,reconstruction)
    print('Length of time series: ', len(data))
    print('Size of compressed file: ',os.path.getsize(args.outfile), 'bytes')
    print('Reconstruction written to: ',reconfile)
elif args.mode == 'd':
    tmpfile = args.infile+'.tmp'
    # extract 7z archive
    subprocess.run(['7z','e',args.infile])
    f_in = open(tmpfile,'rb')
    # read max error from file
    maxerror = np.float32(struct.unpack('f',f_in.read(4))[0])
    # read length of data
    len_data = struct.unpack('I',f_in.read(4))[0]
    # read n from file
    n_nlms = struct.unpack('I',f_in.read(4))[0]
    # initialize quantization (with roughly 65535 bins at the start)
    maxlevel = np.float32(65533*maxerror)
    minlevel = np.float32(-65533*maxerror)
    numbins = int((maxlevel-minlevel)/(2*maxerror))+2
    bins = np.linspace(minlevel,maxlevel,numbins,dtype=np.float32)
    fmtstring = 'H' # 16 bit unsigned
    bin_idx_len = 2 # in bytes
    # initialize predictor
    predictor = NLMS_predictor()
    predictor.init(n_nlms)
    reconstruction = np.zeros(len_data,dtype=np.uint32)
    for i in range(len_data):
        predval = np.float32(predictor.predict(reconstruction,i))
        bin_idx = struct.unpack(fmtstring,f_in.read(bin_idx_len))[0]
        if bin_idx == numbins:
            reconstruction[i] = np.float32(struct.unpack('f',f_in.read(4))[0])
        else:
            reconstruction[i] = predval + bins[bin_idx]
    os.remove(tmpfile)
    # save reconstruction to a file 
    np.save(args.outfile,reconstruction)
    print('Length of time series: ', len_data)
else:
    raise RuntimeError('invalid mode (c and d are the only valid modes)')
