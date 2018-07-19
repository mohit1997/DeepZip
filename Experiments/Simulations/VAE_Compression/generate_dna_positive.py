import numpy as np
import os

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


def generate_positive(rows=400000, file='../../../reference.npy', word=6, length=20):
	data = np.load(file)
	window = length*word
	stride = length*word
	data = data[:((rows-1)*stride + window)]
	X = load_data(data, stride=stride, window=window).astype('|S1')
	
	X = char_map(X, ['0', '1', '2', '3'], ['a', 'c', 'g', 't'])

	l = [word*i for i in range(1, length)]
	X = np.insert(X, l, ' ', axis=1)
	print(X[0])
	ensure_dir('save')
	tr = int(0.6*rows)
	val = int(0.2*rows)
	ts = int(0.2*rows)
	train_set = X[:tr]
	val_set = X[tr:(tr+val)]
	test_set = X[(tr+val):]
	filename = "save/" + "ptb.train.txt"
	with open(filename,"w") as f:
		f.write("\n".join("".join(map(str, x)) for x in train_set))
	filename = "save/" + "ptb.valid.txt"
	with open(filename,"w") as f:
		f.write("\n".join("".join(map(str, x)) for x in val_set))
	filename = "save/" + "ptb.test.txt"
	with open(filename,"w") as f:
		f.write("\n".join("".join(map(str, x)) for x in test_set))


def main():
	generate_positive(400000)



if __name__ =="__main__":
	main()
