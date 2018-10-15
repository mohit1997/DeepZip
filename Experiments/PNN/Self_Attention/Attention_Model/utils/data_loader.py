#Please create your own dataloader for new datasets of the following type

import torch
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import torch.utils.data as data_utils


import os
import sys

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def load_data(seq, stride=1000, window=50):
    seq = np.squeeze(seq)
    data = strided_app(seq, window, stride)
    X = data[:, :]
    # Y = label * np.ones((X.shape[0], 1))
    return X

def char_map(arr, find, vals):
    for i, j in zip(find, vals):
        np.place(arr, arr==i, j)

    return arr


def generate_positive(rows=400000, file='short.npy', word=1, length=200, full=False, stride=None, ratio=0.8):
    data = np.load(file).reshape(-1)
    # data = data[10000:]
    window = length*word + 1
    if stride is None:
        stride = length*word
    data = data[:((rows-1)*stride + window)]
    dataset = load_data(data, stride=stride, window=window)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    # X = char_map(X, ['0', '1', '2', '3'], ['a', 'c', 'g', 't'])
    if full:
        return X, Y

    rows = X.shape[0]
    tr = int(ratio*rows)
    # val = int(0.2*rows)
    # ts = int(0.0*rows)
    train_set_X = X[:tr]
    val_set_X = X[tr:]
    # print("validation split", val_set_X.shape)
    # test_set_X = X[(tr+val):]
    train_set_Y = Y[:tr]
    val_set_Y = Y[tr:]
    # test_set_Y = Y[(tr+val):]
    return train_set_X, train_set_Y, val_set_X, val_set_Y
 
def load_data_set(type,max_len,vocab_size,batch_size):
    """
        Loads the dataset. Keras Imdb dataset for binary classifcation. Keras reuters dataset for multiclass classification
 
        Args:
            type   : {bool} 0 for binary classification returns imdb dataset. 1 for multiclass classfication return reuters set
            max_len: {int} timesteps used for padding
			vocab_size: {int} size of the vocabulary
			batch_size: batch_size
        Returns:
            train_loader: {torch.Dataloader} train dataloader
            x_test_pad  : padded tokenized test_data for cross validating
			y_test      : y_test
            word_to_id  : {dict} words mapped to indices
 
      
        """
   
    INDEX_FROM=3
    if type == 0:
        NUM_WORDS=vocab_size # only use top 1000 words
           # word index offset
 
        train_set,test_set = imdb.load_data(nb_words=NUM_WORDS, index_from=INDEX_FROM)

        x_train,y_train = train_set[0],train_set[1]
        x_test,y_test = test_set[0],test_set[1]
        word_to_id = imdb.get_word_index()
        word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
 
        id_to_word = {value:key for key,value in word_to_id.items()}
        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        n_train = x.shape[0] - 1000
        n_valid = 1000
 
        x_train = x[:n_train]
        y_train = y[:n_train]
        x_test = x[n_train:n_train+n_valid]
        y_test = y[n_train:n_train+n_valid]
 
        print(x_train.shape, y_train.shape)
        print(y_train[0:100])

        #embeddings = load_glove_embeddings("../../GloVe/glove.6B.50d.txt",word_to_id,50)
        x_train_pad = pad_sequences(x_train,maxlen=max_len)
        x_test_pad = pad_sequences(x_test,maxlen=max_len)
        print("Shape of dataset", np.shape(x_train_pad))
        print(x_train_pad[0])
 
        train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.DoubleTensor))
        train_loader = data_utils.DataLoader(train_data,batch_size=batch_size,drop_last=True)

        val_data = data_utils.TensorDataset(torch.from_numpy(x_test_pad).type(torch.LongTensor),torch.from_numpy(y_test).type(torch.DoubleTensor))
        val_loader = data_utils.DataLoader(val_data,batch_size=batch_size,drop_last=True)        
        return train_loader, val_loader, x_test_pad,y_test,word_to_id
    
    elif type=='dna':
        print("loaded dna data")
        word_to_id = {}
        word_to_id['a'] = 0
        word_to_id['c'] = 1
        word_to_id['g'] = 2
        word_to_id['t'] = 3
        word_to_id['n'] = 4
        X_train, Y_train = generate_positive(300000000, file='chr1.npy', length=max_len, stride=1, full=True)
        # X_val, Y_val = generate_positive(50000000, length=max_len, full=True, stride=1)
        # X_val, Y_val = generate_positive(5000000, file='chr1.npy', length=max_len, stride=1, full=True)
        # X_val, Y_val = X_train[::10], Y_train[::10]
        X_val, Y_val = X_train, Y_train
        # print(X_train.shape, X_val.shape)

        train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),torch.from_numpy(Y_train).type(torch.DoubleTensor))
        train_loader = data_utils.DataLoader(train_data,batch_size=batch_size, shuffle=True, drop_last=True)  

        val_data = data_utils.TensorDataset(torch.from_numpy(X_val).type(torch.LongTensor),torch.from_numpy(Y_val).type(torch.DoubleTensor))
        val_loader = data_utils.DataLoader(val_data,batch_size=batch_size,drop_last=True, shuffle=True)        

        return train_loader, val_loader, X_val, word_to_id

    else:
        from keras.datasets import reuters
 
        train_set,test_set = reuters.load_data(path="reuters.npz",nb_words=vocab_size,skip_top=0,index_from=INDEX_FROM)
        x_train,y_train = train_set[0],train_set[1]
        y_train = np.array(y_train)
        print(y_train[0:100])
        x_test,y_test = test_set[0],test_set[1]
        word_to_id = reuters.get_word_index(path="reuters_word_index.json")
        word_to_id = {k:(v+3) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        word_to_id['<EOS>'] = 3
        id_to_word = {value:key for key,value in word_to_id.items()}
        x_train_pad = pad_sequences(x_train,maxlen=max_len)
        x_test_pad = pad_sequences(x_test,maxlen=max_len)
 
 
        train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.LongTensor))
        train_loader = data_utils.DataLoader(train_data,batch_size=batch_size,drop_last=True)
        return train_loader,train_set,test_set,x_test_pad,word_to_id