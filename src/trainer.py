from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger

import models

tf.set_random_seed(42)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store', default=None,
                    dest='train',
                    help='choose training sequence file')
parser.add_argument('-val', action='store', default=None,
                    dest='val',
                    help='choose validation sequence file')
parser.add_argument('-name', action='store', default="model1",
                    dest='name',
                    help='weights will be stored with this name')
parser.add_argument('-model_name', action='store', default=None,
                    dest='model_name',
                    help='name of the model to call')
parser.add_argument('-log_file', action='store',
                    dest='log_file',
                    help='Log file')

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def generate_data(file_path,time_steps):
        series = np.load(file_path)
        series = series.reshape(-1, 1)

        series = series.reshape(-1)

        data = strided_app(series, time_steps+1, 1)

        X = data[:, :-1]
        Y = data[:, -1:]
        
        return X,Y

        
def fit_model(X_train, Y_train, X_val, Y_val, nb_epoch, model):
        model.compile(loss='mean_squared_error', optimizer='adam')
        checkpoint = ModelCheckpoint(arguments.name, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False)
        csv_logger = CSVLogger(arguments.log_file, append=True, separator=';')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=3, verbose=1)

        callbacks_list = [checkpoint, csv_logger, early_stopping]
        #callbacks_list = [checkpoint, csv_logger]
        model.fit(X_train, Y_train, epochs=nb_epoch, verbose=1, shuffle=True, callbacks=callbacks_list, validation_data = (X_val,Y_val))
 
 
arguments = parser.parse_args()
print(arguments)

sequence_length=4
hidden_layer_size=10
num_epochs=20

X_train,Y_train = generate_single_output_data(arguments.train, sequence_length)
print(X_train)
print(Y_train)
X_val,Y_val = generate_single_output_data(arguments.val, sequence_length)
model = getattr(models, arguments.model_name)(sequence_length,hidden_layer_size)
fit_model(X_train, Y_train, X_val, Y_val, num_epochs, model)
