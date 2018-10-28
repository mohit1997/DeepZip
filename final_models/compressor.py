# 
# Compression application using adaptive arithmetic coding
# 
# Usage: python adaptive-arithmetic-compress.py InputFile OutputFile
# Then use the corresponding adaptive-arithmetic-decompress.py application to recreate the original input file.
# Note that the application starts with a flat frequency table of 257 symbols (all set to a frequency of 1),
# and updates it after each byte encoded. The corresponding decompressor program also starts with a flat
# frequency table and updates it after each byte decoded. It is by design that the compressor and
# decompressor have synchronized states, so that the data can be decompressed properly.
# 
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
#
 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM, Flatten, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import contextlib
import arithmeticcoding_fast
import json
from tqdm import tqdm
import struct
import models
import tempfile
import shutil

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model_weights_file',
                    help='model file')
parser.add_argument('-model_name', action='store', dest='model_name',
                    help='model file')
parser.add_argument('-batch_size', action='store', dest='batch_size', type=int,
                    help='model file')
parser.add_argument('-data', action='store', dest='sequence_npy_file',
                    help='data file')
parser.add_argument('-data_params', action='store', dest='params_file',
                    help='params file')
parser.add_argument('-output', action='store',dest='output_file_prefix',
                    help='compressed file name')

args = parser.parse_args()
from keras import backend as K




def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def predict_lstm(X, y, y_original, timesteps, bs, alphabet_size, model_name, final_step=False):
        model = getattr(models, model_name)(bs, timesteps, alphabet_size)
        model.load_weights(args.model_weights_file)
        
        if not final_step:
                num_iters = int((len(X)+timesteps)/bs)
                ind = np.array(range(bs))*num_iters
                
                # open compressed files and compress first few characters using
                # uniform distribution
                f = [open(args.temp_file_prefix+'.'+str(i),'wb') for i in range(bs)]
                bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
                enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]
                prob = np.ones(alphabet_size)/alphabet_size
                cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
                cumul[1:] = np.cumsum(prob*10000000 + 1)        
                for i in range(bs):
                        for j in range(min(timesteps, num_iters)):
                                enc[i].write(cumul, X[ind[i],j])
                cumul = np.zeros((bs, alphabet_size+1), dtype = np.uint64)
                for j in (range(num_iters - timesteps)):
                        prob = model.predict(X[ind,:], batch_size=bs)
                        cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
                        for i in range(bs):
                                enc[i].write(cumul[i,:], y_original[ind[i]])
                        ind = ind + 1
                # close files
                for i in range(bs):
                        enc[i].finish()
                        bitout[i].close()
                        f[i].close()            
        else:
                f = open(args.temp_file_prefix+'.last','wb')
                bitout = arithmeticcoding_fast.BitOutputStream(f)
                enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
                prob = np.ones(alphabet_size)/alphabet_size
                cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
                cumul[1:] = np.cumsum(prob*10000000 + 1)        

                for j in range(timesteps):
                        enc.write(cumul, X[0,j])
                for i in (range(len(X))):
                        prob = model.predict(X[i,:].reshape(1,-1), batch_size=1)
                        cumul[1:] = np.cumsum(prob*10000000 + 1)
                        enc.write(cumul, y_original[i][0])
                enc.finish()
                bitout.close()
                f.close()
        return


# variable length integer encoding http://www.codecodex.com/wiki/Variable-Length_Integers
def var_int_encode(byte_str_len, f):
        while True:
                this_byte = byte_str_len&127
                byte_str_len >>= 7
                if byte_str_len == 0:
                        f.write(struct.pack('B',this_byte))
                        break
                f.write(struct.pack('B',this_byte|128))
                byte_str_len -= 1

def main():
        args.temp_dir = tempfile.mkdtemp()
        args.temp_file_prefix = args.temp_dir + "/compressed"
        tf.set_random_seed(42)
        np.random.seed(0)
        series = np.load(args.sequence_npy_file)
        series = series.reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit(series)

        batch_size = args.batch_size
        timesteps = 64
         
        
        with open(args.params_file, 'r') as f:
                params = json.load(f)

        params['len_series'] = len(series)
        params['bs'] = batch_size
        params['timesteps'] = timesteps

        with open(args.output_file_prefix+'.params','w') as f:
                json.dump(params, f, indent=4)

        alphabet_size = len(params['id2char_dict'])

        series = series.reshape(-1)
        data = strided_app(series, timesteps+1, 1)

        X = data[:, :-1]
        Y_original = data[:, -1:]
        Y = onehot_encoder.transform(Y_original)

        l = int(len(series)/batch_size)*batch_size
        
        predict_lstm(X, Y, Y_original, timesteps, batch_size, alphabet_size, args.model_name)
        if l < len(series)-timesteps:
                predict_lstm(X[l:,:], Y[l:,:], Y_original[l:], timesteps, 1, alphabet_size, args.model_name, final_step = True)
        else:
                f = open(args.temp_file_prefix+'.last','wb')
                bitout = arithmeticcoding_fast.BitOutputStream(f)
                enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout) 
                prob = np.ones(alphabet_size)/alphabet_size
                
                cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
                cumul[1:] = np.cumsum(prob*10000000 + 1)        
                for j in range(l, len(series)):
                        enc.write(cumul, series[j])
                enc.finish()
                bitout.close() 
                f.close()
        
        
        # combine files into one file
        f = open(args.output_file_prefix+'.combined','wb')
        for i in range(batch_size):
                f_in = open(args.temp_file_prefix+'.'+str(i),'rb')
                byte_str = f_in.read()
                byte_str_len = len(byte_str)
                var_int_encode(byte_str_len, f)
                f.write(byte_str)
                f_in.close()
        f_in = open(args.temp_file_prefix+'.last','rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
        f.close()
        shutil.rmtree(args.temp_dir)

                                        
if __name__ == "__main__":
        main()

