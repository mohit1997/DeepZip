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
import arithmeticcoding_shubham_fast as arithmeticcoding_fast
import json
from tqdm import tqdm

### Input/output file names. TODO: use argparse for this
model_weights_file = 'model.h5'
sequence_npy_file = 'short.npy'
input_dir = 'compress_dir'
input_file_prefix = input_dir + '/compressed_file'

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def create_data(rows, p=0.5):
	data = np.random.choice(2, rows, p=[p, 1-p])
	print(np.sum(data))/np.float32(len(data))
	return data
 
def predict_lstm(len_series, timesteps, bs, final_step=False):
	old_model = load_model(model_weights_file)
	wts = old_model.get_weights()
	alphabet_size = 5
	model = Sequential()
	model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, timesteps)))
	model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
	model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
	# model.add(LSTM(128, stateful=False, return_sequences=True))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	# model.add(Activation('tanh'))
	# model.add(Dense(10, activation='relu'))
	# model.add(BatchNormalization())
	model.add(Dense(alphabet_size, activation='softmax'))
#	optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optim)
	model.set_weights(wts)
	
	if not final_step:
		num_iters = int((len_series)/bs)
		series_2d = np.zeros((bs,num_iters), dtype = np.uint8)
		# open compressed files and decompress first few characters using
		# uniform distribution
		f = [open(input_file_prefix+'.'+str(i),'rb') for i in range(bs)]
		bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
		dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]
#		freqs = [arithmeticcoding.SimpleFrequencyTable(np.ones(alphabet_size)) for i in range(bs)]
		prob = np.ones(alphabet_size)/alphabet_size
		cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
		cumul[1:] = np.cumsum(prob*10000000 + 1)		
		for i in range(bs):
			for j in range(min(num_iters,timesteps)):
				series_2d[i,j] = dec[i].read(cumul, alphabet_size)
		cumul = np.zeros((bs, alphabet_size+1), dtype = np.uint64)
		for j in tqdm(range(num_iters - timesteps)):
			prob = model.predict(series_2d[:,j:j+timesteps], batch_size=bs)
			cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
			for i in range(bs):
#				series_2d[i,j+timesteps] = arithmetic_step(prob[i,:], freqs[i], dec[i])
				series_2d[i,j+timesteps] = dec[i].read(cumul[i,:], alphabet_size)
		# close files
		for i in range(bs):
			bitin[i].close()
			f[i].close()
		return series_2d.reshape(-1)
	else:
		series = np.zeros(len_series, dtype = np.uint8)
		f = open(input_file_prefix+'.last','rb')
		bitin = arithmeticcoding_fast.BitInputStream(f)
		dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
		prob = np.ones(alphabet_size)/alphabet_size
 
		cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
		cumul[1:] = np.cumsum(prob*10000000 + 1)		
		for j in range(min(timesteps,len_series)):
			series[j] = dec.read(cumul, alphabet_size)
		for i in tqdm(range(len_series-timesteps)):
			prob = model.predict(series[i:i+timesteps].reshape(1,-1), batch_size=1)
			cumul[1:] = np.cumsum(prob*10000000 + 1)
			series[i+timesteps] = dec.read(cumul, alphabet_size)
		bitin.close()
		f.close()
		return series

def arithmetic_step(prob, freqs, dec):
	freqs.update_table(prob*10000000+1)
	return dec.read(freqs)

def main():
	tf.set_random_seed(42)
	np.random.seed(0)
	f = open(input_file_prefix+'.params','r')
	param_dict = json.loads(f.read())
	f.close()
	len_series = param_dict['len_series']
	batch_size = param_dict['bs']
	timesteps = param_dict['timesteps']

	series = np.zeros(len_series,dtype=np.uint8)

	l = int(len_series/batch_size)*batch_size
	series[:l] = predict_lstm(l, timesteps, batch_size)
	if l < len_series:
		series[l:] = predict_lstm(len_series - l, timesteps, 1, final_step = True)
	
	f = open('temp_2','w')
	f.write(''.join([str(s) for s in series]))
	f.close()

if __name__ == "__main__":
	main()

#np.random.seed(0)
#
#def softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    e_x = np.exp(x)
#    return e_x
#
## Command line main application function.
#
#def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
#    nrows = ((a.size - L) // S) + 1
#    n = a.strides[0]
#    return np.lib.stride_tricks.as_strided(
#        a, shape=(nrows, L), strides=(S * n, n), writeable=False)
#
#
#def generate_probability(n_classes):
#	return softmax(np.random.uniform(10, 11, n_classes))
#
#
#def main(args):
#	# Handle command line arguments
#	if len(args) != 1:
#		sys.exit("Usage: python adaptive-arithmetic-compress.py OutputFile")
#	outputfile,  = args
#	
#	# Perform file compression
#	data = np.load('chr1.npy').astype(np.int32)
#	with contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) as bitout:
#		compress(data, bitout)
#
#	
#
#
#
#
#
#def compress(data, bitout):
#	print(bitout)
#	print(data.shape, np.unique(data))
#	data = data[:10000]
#	probs = np.load('prob_temp.npy').astype(np.float32)[:10000]
#
#	# from sklearn import preprocessing
#	# le = preprocessing.LabelEncoder()
#	# initfreqs = arithmeticcoding.FlatFrequencyTable(6)
#	# freqs = arithmeticcoding.SimpleFrequencyTable([500, 500, 500, 500, 100, 1])
#	enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
#
#	for i in range(len(data)):
#		symbol = data[i]
#		prob = probs[i]
#		l = [int(p*10000000+1) for p in prob]
#		l.append(1)
#		# print(prob, symbol)
#		freqs = arithmeticcoding.SimpleFrequencyTable(l)
#		enc.write(freqs, symbol[0])
#
#	enc.write(freqs, 5)
#	enc.finish()
#
#	# sym_list = np.array([97, 99, 103, 110, 116])
#	# le.fit(sym_list)
#	# while True:
#	# 	# Read and encode one byte
#	# 	symbol = inp.read(1)
#
#	# 	if len(symbol) == 0:
#	# 		break
#	# 	symbol = symbol[0] if python3 else ord(symbol)
#	# 	o = le.transform([symbol])
#	# 	symbol = o[0]
#	# 	print(symbol)
#	# 	enc.write(freqs, symbol)
#	# 	# freqs.increment(symbol)
#	# print(np.unique(sym_list))
#	# enc.write(freqs, 5)  # EOF
#	# enc.finish()  # Flush remaining code bits
#
#
#
#
#
## Main launcher
#if __name__ == "__main__":
#	main(sys.argv[1 : ])
