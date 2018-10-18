from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten, CuDNNLSTM
from math import sqrt
from keras.layers.embeddings import Embedding
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
tf.set_random_seed(42)


import keras.backend as K
def loss_fn(y_true, y_pred):
    return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)

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
	# model.add(Embedding(y.shape[1], 32, batch_input_shape=(bs, X.shape[1])))
	model.add(CuDNNLSTM(32, batch_input_shape=(bs, X.shape[1], X.shape[2]), stateful=True, return_sequences=False))
	# model.add(CuDNNLSTM(64, stateful=False, return_sequences=False))
	# model.add(CuDNNLSTM(128, stateful=False, return_sequences=True))
	# model.add(Flatten())
	# model.add(Dense(128, activation='relu'))
	# model.add(Activation('tanh'))
	# model.add(Dense(32, activation='relu'))
	# model.add(BatchNormalization())
	model.add(Dense(y.shape[1], activation='softmax'))
	optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, clipnorm=0.05)
	model.compile(loss=loss_fn, optimizer=optim)
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=bs, verbose=1, shuffle=False)
		model.reset_states()
	return model
 

series = np.load('markov_seq.npy')

# series = series[0:100000]
series = series.reshape(-1, 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit(series)

series = series.reshape(-1)

data = strided_app(series, 65, 1)

batch_size = 64

l = int(len(data)/batch_size) * batch_size

data = data[:l] ##selecting ony 9980 points (divisible by batch_size)

X = data[:, :-1]
X_ = X.reshape(-1, 1)
X_ = onehot_encoder.transform(X_)
X_onehot = X_.reshape(X.shape[0], X.shape[1], -1)

# Y = np.logical_xor(data[:, 0:1], data[:, 10:11])*1.0
Y = data[:, -1:]
Y = onehot_encoder.transform(Y)


print(X[0:10, 0], X[0:10, 10])
print(Y[0:10])

for r in range(1):
	# fit the model
    lstm_model = fit_lstm(X_onehot, Y, batch_size, 5, 64)

