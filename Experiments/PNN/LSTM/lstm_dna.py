import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import pickle
import matplotlib.pyplot as plt


tf.set_random_seed(42)
np.random.seed(42)
#Epochs
epochs = 20

time_steps=10
#hidden LSTM units
window=20
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 4 classes(0-3).
n_classes=10
#size of batch
batch_size=16

num_units = [128, 32]


#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
# input=tf.unstack(x ,time_steps,1)

def ensure_dir(directory):
    if not os.path.exists(directory):
		os.makedirs(directory)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
    	# if(start_idx + batchsize >= inputs.shape[0]):
    	# 	break;

        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def fc(x, h_in, h_out, relu=False):
	W = weight_variable([h_in, h_out])
	b = bias_variable([h_out])

	y = tf.matmul(x, W) + b
	if relu:
		return tf.nn.relu(y)
	return y


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def generate(array, encoder, lag=10, window=window, steps=time_steps):
	data = strided_app(array, window, 1)
	l = [np.roll(data, -i*lag, axis=0) for i in range(steps)]
	l = [data] + l
	X = l[:-1]
	Y = l[-1][:, -1:]
	Y = encoder.transform(Y)
	X = np.stack(X, axis=1)
	return X, Y

def nn(x):
	cells = [rnn.BasicLSTMCell(num_units=n) for n in num_units]
	stacked_rnn_cell = rnn.MultiRNNCell(cells)
	o1,_=rnn.static_rnn(stacked_rnn_cell,x,dtype="float32")

	o = fc(o1[-1], num_units[-1], 4)

	return o

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['Train_loss']))

    y1 = hist['Train_loss']
    y2 = hist['Val_loss']

    plt.plot(x, y1, label='train')
    plt.plot(x, y2, label='val')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()




def main():
	print("hello")
	series = np.load('../short.npy')
	print(np.max(series))
	train = series[0:20000]
	train = train.reshape(-1, 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	onehot_encoded = onehot_encoder.fit(train)
	X, Y = generate(train, encoder=onehot_encoder)
	X_val, Y_val = generate(series[20000:30000], encoder=onehot_encoder)
	print(Y.shape)
	print(X.shape)
	print(series.shape)

	#loss_function
	x = tf.placeholder("float",[None,time_steps,window])
	input = tf.unstack(x ,time_steps,1)
	#input label placeholder
	y_ = tf.placeholder("float",[None,Y.shape[1]])

	prediction = nn(input)


	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
	#optimization
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	#model evaluation
	correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	#initialize variables
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	train_hist = {}
	train_hist['Train_loss'] = []
	train_hist['Val_loss'] = []
	train_hist['Train_acc'] = []
	train_hist['Val_acc'] = []

	sess.run(tf.global_variables_initializer())
	for i in range(epochs):
		# print(X.shape)
		# print(X.dtype)
		l = []
		a = []
		for batch_x, batch_y  in iterate_minibatches(X, Y, batchsize=batch_size, shuffle=True):
			lis = sess.run([optimizer, loss, accuracy], feed_dict={x: batch_x, y_: batch_y})
			l.append(lis[1])
			a.append(lis[2])
		if i%1 == 0:
			print("Epoch", i)
			print("Training Error and Accuracy", np.mean(l), np.mean(a))
			train_hist['Train_loss'].append(np.mean(l))
			train_hist['Train_acc'].append(np.mean(a))

			lis = sess.run([loss, accuracy], feed_dict={x: X_val, y_: Y_val})
			print("Validation Error and Accuracy", lis[0], lis[1])
			train_hist['Val_loss'].append(lis[0])
			train_hist['Val_acc'].append(lis[1])
	
	ensure_dir('results')
	with open('results/train_hist.pkl', 'wb') as f:
		pickle.dump(train_hist, f)
	
	print(train_hist)
	show_train_hist(train_hist, save=True, path='results/train_hist.png')




if __name__ == "__main__":
	main()