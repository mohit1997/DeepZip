# 
# Compression application using adaptive arithmetic coding
# 
# Usage: python adaptive-arithmetic-compress.py InputFile OutputFile
# Then use the corresponding adaptive-arithmetic-decompress.py application to recreate the original input file.
# Note that the application starts with a flat frequency table of 257 symbols (all set to a frequency of 1),
# and updates it after each byte encoded. The corresponding decompressor program also starts with a flat
# frequency table and updates it after each byte decoded. It is by design that the compressor and
# decompressor have synchronized states, so that the data can be decompressed properly.
# 
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 

import contextlib, sys
import arithmeticcoding
python3 = sys.version_info.major >= 3
import numpy as np

np.random.seed(0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x

# Command line main application function.

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def generate_probability(n_classes):
	return softmax(np.random.uniform(10, 11, n_classes))


def main(args):
	# Handle command line arguments
	if len(args) != 1:
		sys.exit("Usage: python adaptive-arithmetic-compress.py OutputFile")
	outputfile,  = args
	
	# Perform file compression
	data = np.load('chr1.npy').astype(np.int32)
	with contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) as bitout:
		compress(data, bitout)


def compress(data, bitout):
	print(bitout)
	print(data.shape, np.unique(data))
	strided_data = strided_app(data, 30, 1)
	probs = np.load('prob_temp.npy').astype(np.float32)

	# from sklearn import preprocessing
	# le = preprocessing.LabelEncoder()
	# initfreqs = arithmeticcoding.FlatFrequencyTable(6)
	# freqs = arithmeticcoding.SimpleFrequencyTable([500, 500, 500, 500, 100, 1])
	enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

	for i in range(len(data)):
		symbol = data[i]
		prob = probs[i]
		l = [int(p*10000000+1) for p in prob]
		l.append(1)
		# print(prob, symbol)
		freqs = arithmeticcoding.SimpleFrequencyTable(l)
		enc.write(freqs, symbol[0])

	enc.write(freqs, 5)
	enc.finish()

	# sym_list = np.array([97, 99, 103, 110, 116])
	# le.fit(sym_list)
	# while True:
	# 	# Read and encode one byte
	# 	symbol = inp.read(1)

	# 	if len(symbol) == 0:
	# 		break
	# 	symbol = symbol[0] if python3 else ord(symbol)
	# 	o = le.transform([symbol])
	# 	symbol = o[0]
	# 	print(symbol)
	# 	enc.write(freqs, symbol)
	# 	# freqs.increment(symbol)
	# print(np.unique(sym_list))
	# enc.write(freqs, 5)  # EOF
	# enc.finish()  # Flush remaining code bits





# Main launcher
if __name__ == "__main__":
	main(sys.argv[1 : ])
