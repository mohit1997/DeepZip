import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from time import ctime
import os
import pickle
import matplotlib.pyplot as plt
save_path = "Models/"

window = 1000
learning_rate = 1e-3
batch_size = 128
forward = 10
tf.set_random_seed(42)


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


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def load_data(seq, stride=1000):
	seq = np.squeeze(seq)
	data = strided_app(seq, window + forward, stride)
	X = data[:, :-forward]
	Y = data[:, -forward:]
	re = Y.reshape(-1, 1)
	X = np.expand_dims(X, -1)
	onehot_encoder = OneHotEncoder(sparse=False)
	onehot_encoded = onehot_encoder.fit(np.expand_dims(seq, -1))
	re = onehot_encoder.transform(re)
	Y = re.reshape(Y.shape[0], Y.shape[1], re.shape[1]) 
	return X, Y	


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


def predict(inp, isTrain):
	h_conv3 = tf.layers.conv1d(inp, filters=16, kernel_size=(1), strides=1, padding='SAME', activation=tf.nn.relu)
	b_3 = tf.layers.batch_normalization(h_conv3, axis=-1, training=isTrain)
	# p_3 = tf.layers.max_pooling1d(b_3, pool_size=[2], strides=[2])

	flatten = tf.contrib.layers.flatten(b_3)

	shp = flatten.get_shape().as_list()
	# print(shp)
	h1 = fc(flatten, shp[1], 100, relu=True)
	b_fc1 = tf.layers.batch_normalization(h1, axis=-1, training=isTrain)
	h2 = fc(b_fc1, 100, 100)
	b_fc2 = tf.layers.batch_normalization(h2, axis=-1, training=isTrain)
	h3 = fc(b_fc2, 100, 4)
	return h3

def nn(x, isTrain):# x - input_placeholder, y - ouput_placeholder
	# inp = tf.expand_dims(x, 2)
	inp = x

	h_conv1 = tf.layers.conv1d(inp, filters=16, kernel_size=(10), strides=4, padding='SAME', activation=tf.nn.relu)
	b_1 = tf.layers.batch_normalization(h_conv1, axis=-1, training=isTrain)
	# p_1 = tf.layers.max_pooling1d(b_1, pool_size=[4], strides=[4])

	h_conv2 = tf.layers.conv1d(b_1, filters=32, kernel_size=(5), strides=4, padding='SAME', activation=tf.nn.relu)
	b_2 = tf.layers.batch_normalization(h_conv2, axis=-1, training=isTrain)
	# p_2 = tf.layers.max_pooling1d(b_2, pool_size=[4], strides=[4])

	#p_2 = p_1

	h_conv3 = tf.layers.conv1d(b_2, filters=64, kernel_size=(3), strides=4, padding='SAME', activation=tf.nn.relu)
	b_3 = tf.layers.batch_normalization(h_conv3, axis=-1, training=isTrain)
	# p_3 = tf.layers.max_pooling1d(b_3, pool_size=[4], strides=[4])
	
	

	o = [predict(b_3, isTrain) for i in range(forward)]
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
	print("Hello")

	# seq = np.load('integers.npy')	
	series = np.load('reference.npy')
	print(np.max(series))
	print(len(series))
	seq = series[:1000000]
	seq = seq.reshape(-1, 1)
	test = series[1000000:1050000]

	X, Y = load_data(seq, stride=50)
	X1, Y1 = load_data(test)

	print(X.shape, Y.shape)

	x = tf.placeholder(tf.float32, [None, window, 1])
	isTrain = tf.placeholder(dtype=tf.bool)
	y_ = tf.placeholder(tf.float32, [None, Y.shape[1], Y.shape[2]])
	y_list = tf.unstack(y_, Y.shape[1], axis=1)


	y = nn(x, isTrain)
	losses = []
	bce = 0



	for i in range(len(y_list)):
		l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_list[i], logits=y[i]))
		bce += l
		losses.append(l)

	loss = bce
	optimizer = tf.train.AdamOptimizer(learning_rate)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		gradients, variables = zip(*optimizer.compute_gradients(loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		optimize = optimizer.apply_gradients(zip(gradients, variables))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	saver = tf.train.Saver()

	sess.run(tf.global_variables_initializer())

	train_hist = {}
	train_hist['Train_loss'] = []
	train_hist['Val_loss'] = []
	train_hist['Train_acc'] = []
	train_hist['Val_acc'] = []

	for i in range(20):
		# print(X.shape)
		# print(X.dtype)		
		counter = 0
		for batch_x, batch_y in iterate_minibatches(X, Y, batchsize=batch_size, shuffle=True):
			sess.run(optimize, feed_dict={x: batch_x, y_: batch_y, isTrain: True})
			counter += 1
			if counter%1000 == 0:
				print(counter)

		if(i%1 == 0):
			print("Epoch", i)
			l = sess.run([losses], feed_dict={x: X[::10], y_: Y[::10], isTrain: True})
			print("Train Error", np.mean(l))
			train_hist['Train_loss'].append(np.mean(l))
			
			l1 = sess.run([losses], feed_dict={x: X1, y_: Y1, isTrain: False})
			print("Test Error", np.mean(l1))
			train_hist['Val_loss'].append(np.mean(l1))
			model_name = save_path + "model" + ".ckpt"
			path = saver.save(sess, model_name)


	ensure_dir('results')
	with open('results/train_hist.pkl', 'wb') as f:
		pickle.dump(train_hist, f)
	
	print(train_hist)
	show_train_hist(train_hist, save=True, path='results/train_hist.png')

if __name__ == "__main__":
	main()
