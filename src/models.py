from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, LSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from math import sqrt
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
import tensorflow as tf
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger
from keras import backend as K


def biGRU(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(GRU(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(GRU(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def biGRU_big(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(GRU(128, stateful=False, return_sequences=True)))
        model.add(Bidirectional(GRU(128, stateful=False, return_sequences=False)))
#        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def biGRU_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(GRU(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(GRU(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def biLSTM(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(LSTM(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(LSTM(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model


def biLSTM_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(LSTM(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(LSTM(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_big(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 64, batch_input_shape=(bs, time_steps)))
        model.add(LSTM(64, stateful=False, return_sequences=True))
        model.add(LSTM(64, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_bn(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_selu(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation=keras.activations.selu, kernel_initializer=init))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_selu_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(LSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        init = keras.initializers.lecun_uniform(seed=0)
        model.add(Dense(64, activation=keras.activations.selu, kernel_initializer=init))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def GRU_multi(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(GRU(32, stateful=False, return_sequences=True))
        model.add(GRU(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def GRU_multi_big(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(GRU(128, stateful=False, return_sequences=True))
        model.add(GRU(128, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def GRU_multi_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(GRU(32, stateful=False, return_sequences=True))
        model.add(GRU(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def FC_4layer_16bit(bs,time_steps, alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 5, batch_input_shape=(bs, time_steps)))
        model.add(Flatten())
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def FC_4layer(bs,time_steps, alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 5, batch_input_shape=(bs, time_steps)))
        model.add(Flatten())
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def FC_4layer_big(bs,time_steps, alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Flatten())
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def FC_16bit(bs,time_steps,alphabet_size):
        k.set_floatx('float16')
        model = Sequential()
        init = keras.initializers.lecun_uniform(seed=0)
        model.add(embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(flatten())
        model.add(dense(1024, activation='relu', kernel_initializer=init))
        model.add(dense(64, activation='relu', kernel_initializer=init))
        model.add(dense(alphabet_size, activation='softmax'))
        return model


def FC(bs,time_steps,alphabet_size):
        model = Sequential()
        init = keras.initializers.lecun_uniform(seed=0)
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_initializer=init))
        model.add(Dense(64, activation='relu', kernel_initializer=init))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

# models for Siemens floating point data compression
def FC_siemens_1hiddenlayer(input_dim,hidden_layer_size):
    model = Sequential()
    model.add(Dense(hidden_layer_size,activation='relu',input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dense(1))
    return model
