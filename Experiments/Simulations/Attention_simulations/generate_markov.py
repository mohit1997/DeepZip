import numpy as np
np.random.seed(0)


def generate_and_save(seqlength, p):
	seq = np.random.choice(2, (50))
	seq = list(seq)
	for i in range(1000000):
		seq.append(int(np.logical_xor(seq[-1], seq[-50])))
	seq= np.array(seq)

	np.save('markovseq', seq)


if __name__ == "__main__":
	generate_and_save(1000000, 0.1)