from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten, CuDNNLSTM
from math import sqrt
from keras.layers.embeddings import Embedding
# from matplotlib import pyplot
from keras.models import load_model
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
tf.set_random_seed(42)

import os

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
def fit_lstm(X, Y, bs, initial=False):
	y = Y
	old_model = load_model('model.h5')
	wts = old_model.get_weights()
	model = Sequential()
	model.add(Embedding(y.shape[1], 32, batch_input_shape=(bs, X.shape[1])))
	model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
	model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
	# model.add(LSTM(128, stateful=False, return_sequences=True))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	# model.add(Activation('tanh'))
	# model.add(Dense(10, activation='relu'))
	# model.add(BatchNormalization())
	model.add(Dense(y.shape[1], activation='softmax'))
	optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optim)
	model.set_weights(wts)
	print(y.shape)
	o = model.predict(X, batch_size=bs, verbose=1)
	if initial:
		temp = 1.0/5*np.ones((X.shape[1], y.shape[-1]))
		o = np.concatenate([temp, o], axis=0)
	return o
 

series = np.load('short.npy')

# series = series[0:100000]
series = series.reshape(-1, 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit(series)

series = series.reshape(-1)

data = strided_app(series, 61, 1)

batch_size = 10000

l = int(len(data)/batch_size) * batch_size
X = data[:, :-1]
Y = data[:, -1:]
Y = onehot_encoder.transform(Y)


X1 = X[:l]
Y1 = Y[:l]
o1 = fit_lstm(X1, Y1, batch_size, initial=True)

if l < len(X):

	X2 = X[l:]
	Y2 = Y[l:]
	o2 = fit_lstm(X2, Y2, 1)

	o = np.concatenate([o1, o2], axis=0)
	np.save('prob_temp', o)
else:
	np.save('prob_temp', o1)

