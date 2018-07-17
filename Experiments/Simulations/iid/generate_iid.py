import numpy as np
import os.path
np.random.seed(42)
prob = 'prob.npy'
vals_file = 'vals.npy'

def generate(file='../reference.npy', size=1000):
	if os.path.isfile(prob) and os.path.isfile(vals_file):
		l = np.load(prob)
		vals = np.load(vals_file)
	else:
		data = np.load(file)
		data= data[:]
		print(data.shape)
		vals = np.unique(data)
		print(vals)
		l = []
		total = 0
		for i in vals:
			count = np.sum(1.0*(data == i))
			l.append(count)
			total = total + count
			# print(i, count)
		l = np.array(l/total)
		np.save('prob', l)
		np.save('vals', vals)

	output = np.random.choice(vals, size=size, p=l)
	# print(output)
	return output





def main():
	generate()

if __name__ == "__main__":
	main()