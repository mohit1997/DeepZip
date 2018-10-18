import numpy as np

def generate_markov(seq_length):
	a = np.random.choice(2, 60)
	l = list(a)
	for _ in range(seq_length-60):
		l.append(np.logical_xor(l[-40], l[-1]))

	a = np.array(l)
	np.save('markov_seq', a)

if __name__ == "__main__":
	generate_markov(1000000)