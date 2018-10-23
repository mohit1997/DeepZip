from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from math import sqrt
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger


tf.set_random_seed(42)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-data', action='store', default="markov_seq.npy",
                    dest='data',
                    help='choose sequence file')
parser.add_argument('-gpu', action='store', default="0",
                    dest='gpu',
                    help='choose gpu number')
parser.add_argument('-model', action='store', default="model1",
                    dest='name',
                    help='weights will be stored with this name')
parser.add_argument('-epochs', action='store', default=2,
                    dest='nbepochs', type=int,
                    help='weights will be stored with this name')
parser.add_argument('-len', action='store', default=64,
                    dest='length',
                    help='Truncated Length', type=int)


import keras.backend as K

def loss_fn(y_true, y_pred):
    return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)




def create_data(rows, p=0.5):
	data = np.random.choice(2, rows, p=[p, 1-p])
	print(np.sum(data))/np.float32(len(data))
	return data
 
# fit an LSTM network to training data
def fit_lstm(X, Y, bs, nb_epoch, neurons):
	y = Y
	decay = bs*1.0/len(X)
	model = Sequential()
	model.add(Embedding(y.shape[1], 32, batch_input_shape=(bs, X.shape[1])))
	model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=True)))
	model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
	# model.add(LSTM(128, stateful=False, return_sequences=True))
	# decay = bs*1.0/len(X)
	# model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	# model.add(Activation('tanh'))
	# model.add(Dense(10, activation='relu'))
	# model.add(BatchNormalization())
	model.add(Dense(y.shape[1], activation='softmax'))
	optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
	model.compile(loss=loss_fn, optimizer=optim)
	filepath = arguments.name
	logfile = arguments.name + '.log.csv'
	csv_logger = CSVLogger(logfile, append=True, separator=';')
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
	callbacks_list = [checkpoint, csv_logger]
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=bs, verbose=1, shuffle=True, callbacks=callbacks_list)
		model.reset_states()
	return model
 
arguments = parser.parse_args()
print(arguments)
os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
series = np.load(arguments.data)
print(series.shape)

# series = series[0:100000]
series = series.reshape(-1, 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit(series)

series = series.reshape(-1)

data = strided_app(series, arguments.length+1, 1)

batch_size = 1024

l = int(len(data)/batch_size) * batch_size

data = data[:l] ##selecting ony 9980 points (divisible by batch_size)

X = data[:, :-1]
# Y = np.logical_xor(data[:, 0:1], data[:, 10:11])*1.0
Y = data[:, -1:]
Y = onehot_encoder.transform(Y)


# print(X[20000:20010], X[20000:20010])
# print(Y[20000:20010])

for r in range(1):
	# fit the model
    lstm_model = fit_lstm(X, Y, batch_size, args.nbepochs, 32)

