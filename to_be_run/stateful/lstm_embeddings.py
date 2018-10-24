from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, TimeDistributed
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
parser.add_argument('-d', action='store', default="chr20_hg19.npy",
                    dest='data',
                    help='choose sequence file')
parser.add_argument('-gpu', action='store', default="0",
                    dest='gpu',
                    help='choose gpu number')
parser.add_argument('-name', action='store', default="model1",
                    dest='name',
                    help='weights will be stored with this name')
parser.add_argument('-len', action='store', default=64,
                    dest='length',
                    help='Truncated Length', type=int)
parser.add_argument('-clip', action='store', default=0.05,
                    dest='clip_value',
                    help='Clip Norm paramter', type=float)

import keras.backend as K

def loss_fn(y_true, y_pred):
    return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)

def loss_fnv2(y_true, y_pred):
	print(y_true, y_pred)
	return 1/np.log(2) * K.sparse_categorical_crossentropy(y_true, y_pred)

def last_loss(y_true,y_pred):
    y_ = y_true[:, -1:]
    y = y_pred[:, -1:]
    return 1/np.log(2) * K.categorical_crossentropy(y_, y)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


np.random.seed(0)

def create_data(rows, p=0.5):
	data = np.random.choice(2, rows, p=[p, 1-p])
	print(np.sum(data))/np.float32(len(data))
	return data
 
# fit an LSTM network to training data
def fit_lstm(X, Y, bs, nb_epoch, neurons):
	y = Y
	model = Sequential()
	model.add(Embedding(y.shape[-1], 32, batch_input_shape=(bs, X.shape[1])))
	# model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
	model.add(CuDNNLSTM(32, stateful=True, return_sequences=True))
	model.add(CuDNNLSTM(32, stateful=True, return_sequences=True))
	# model.add(LSTM(128, stateful=False, return_sequences=True))
	# model.add(Flatten())
	model.add(TimeDistributed(Dense(64, activation='relu')))
	# model.add(Activation('tanh'))
	# model.add(Dense(10, activation='relu'))
	# model.add(BatchNormalization())
	model.add(TimeDistributed(Dense(n_classes, activation='softmax')))

	# model_ = multi_gpu_model(model, gpus=2)

	optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=arguments.clip_value)
	model.compile(loss=loss_fn, optimizer=optim, metrics=[last_loss])
	filepath = arguments.name + "weights{loss:.4f}.best.hdf5"
	logfile = arguments.name + 'log.csv'
	csv_logger = CSVLogger(logfile, append=True, separator=';')
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
	callbacks_list = [checkpoint, csv_logger]
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=bs, verbose=1, shuffle=False, callbacks=callbacks_list)
		model.reset_states()
	return model
 

arguments = parser.parse_args()
print(arguments)
os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
series = np.load(arguments.data)[59000:1000000]
print(series.shape)

series = series.reshape(-1, 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit(series)

series = series.reshape(-1)
n_classes = len(np.unique(series))

data = strided_app(series, arguments.length+1, 1)

batch_size = arguments.length
# batch_size = 64

l = int(len(data)/batch_size) * batch_size

data = data[:l] # (divisible by batch_size)

X = data[:, :-1]
Y = data[:, 1:]
# Y = np.expand_dims(Y, axis=-1)
Y_ = Y.reshape(-1, 1)
Y_ = onehot_encoder.transform(Y_)
Y_onehot = Y_.reshape(Y.shape[0], Y.shape[1], -1)


lstm_model = fit_lstm(X, Y_onehot, batch_size, nb_epoch=5, neurons=32)

