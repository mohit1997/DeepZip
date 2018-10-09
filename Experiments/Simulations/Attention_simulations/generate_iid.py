import numpy as np
np.random.seed(0)


def generate_and_save(seqlength, p):
	seq = np.random.choice(2, seqlength, p=[p, 1-p])
	np.save('binseq', seq)


if __name__ == "__main__":
	generate_and_save(1000000, 0.1)

