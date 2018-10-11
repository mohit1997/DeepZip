from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from math import sqrt
from keras.layers.embeddings import Embedding
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
tf.set_random_seed(42)

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
def fit_lstm(X, Y, batch_size, nb_epoch, neurons):
	y = Y

	model = Sequential()
	model.add(Embedding(2, 8, input_length=X.shape[1]))
	model.add(LSTM(64, batch_input_shape=(batch_size, X.shape[1], 8), stateful=False, return_sequences=False))
	# model.add(LSTM(128, stateful=False, return_sequences=True))
	# model.add(Flatten())
	model.add(Dense(200, activation='tanh'))
	# model.add(Activation('tanh'))
	model.add(Dense(10, activation='tanh'))
	# model.add(BatchNormalization())
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=True)
		model.reset_states()
	return model
 

series = np.load('markov_seq.npy')

series = series[0:10000]
series = series.reshape(-1, 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit(series)

series = series.reshape(-1)

data = strided_app(series, 21, 1)
data = data[:9980] ##selecting ony 9980 points (divisible by batch_size)

X = data[:, :-1]
# Y = np.logical_xor(data[:, 0:1], data[:, 10:11])*1.0
Y = data[:, -1:]
Y = onehot_encoder.transform(Y)


print(X[0:10, 0], X[0:10, 10])
print(Y[0:10])

for r in range(1):
	# fit the model
    lstm_model = fit_lstm(X, Y, 20, 5, 32)

