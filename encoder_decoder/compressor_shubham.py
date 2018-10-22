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
sequence_npy_file = 'chr1.npy'
output_dir = 'compress_dir'
output_file_prefix = output_dir + '/compressed_file'


alphabet_size = 5

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def create_data(rows, p=0.5):
	data = np.random.choice(2, rows, p=[p, 1-p])
	print(np.sum(data))/np.float32(len(data))
	return data
 
def predict_lstm(X, y, y_original, timesteps, bs, final_step=False):
	old_model = load_model(model_weights_file)
	wts = old_model.get_weights()

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
		num_iters = int((len(X)+timesteps)/bs)
		ind = np.array(range(bs))*num_iters
		
		# open compressed files and compress first few characters using
		# uniform distribution
		f = [open(output_file_prefix+'.'+str(i),'wb') for i in range(bs)]
		bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
		enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]
		prob = np.ones(alphabet_size)/alphabet_size
		cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
		cumul[1:] = np.cumsum(prob*10000000 + 1)	
		for i in range(bs):
			for j in range(min(timesteps, num_iters)):
				enc[i].write(cumul, X[ind[i],j])
		cumul = np.zeros((bs, alphabet_size+1), dtype = np.uint64)
		for j in tqdm(range(num_iters - timesteps)):
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
		f = open(output_file_prefix+'.last','wb')
		bitout = arithmeticcoding_fast.BitOutputStream(f)
		enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
		prob = np.ones(alphabet_size)/alphabet_size
		cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
		cumul[1:] = np.cumsum(prob*10000000 + 1)	

		for j in range(timesteps):
			enc.write(cumul, X[0,j])
		for i in tqdm(range(len(X))):
			prob = model.predict(X[i,:].reshape(1,-1), batch_size=1)
			cumul[1:] = np.cumsum(prob*10000000 + 1)
			enc.write(cumul, y_original[i][0])
		enc.finish()
		bitout.close()
		f.close()
	return


def main():
	tf.set_random_seed(42)
	np.random.seed(0)
	series = np.load(sequence_npy_file)
#	series = series[:1000]
	series = series.reshape(-1, 1)
	f = open('temp_1','w')
	f.write(''.join([str(s[0]) for s in series]))
	f.close()
	onehot_encoder = OneHotEncoder(sparse=False)
	onehot_encoded = onehot_encoder.fit(series)

	batch_size = 10000
	timesteps = 60

	series = series.reshape(-1)
	data = strided_app(series, timesteps+1, 1)

	X = data[:, :-1]
	Y_original = data[:, -1:]
	Y = onehot_encoder.transform(Y_original)

	l = int(len(series)/batch_size)*batch_size
	predict_lstm(X, Y, Y_original, timesteps, batch_size)
	if l < len(series)-timesteps:
		predict_lstm(X[l:,:], Y[l:,:], Y_original[l:], timesteps, 1, final_step = True)
	else:
		f = open(output_file_prefix+'.last','wb')
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
	param_dict = {'len_series': len(series), 'bs': batch_size, 'timesteps': timesteps}
	f = open(output_file_prefix+'.params','w')
	f.write(json.dumps(param_dict))
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
