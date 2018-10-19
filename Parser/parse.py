def read_fasta(fp):                
	name, seq = None, []        
	for line in fp:
	    line = line.rstrip()
	    if line.startswith(">"):
	        if name: yield (name, ''.join(seq))
	        name, seq = line, []
	    else:
	        seq.append(line)
	if name: yield (name, ''.join(seq))


## set range to choose which chromosomes files to parse
## edit chromosome_list appropriately
chromosomes_list = [1]
for i in chromosomes_list:
	filename = 'fastq_files/chr' + str(i) + '.fa'

	names = []
	seqs = []

	with open(filename) as fp:
		for name, seq in read_fasta(fp):
			names.append(name)
			seqs.append(seq)

	# print(len(seqs))
	a = [len(j) for j in seqs]
	print(a)

	# import pickle

	# with open('seqs', 'wb') as fp:
	#     pickle.dump(seqs[:025], fp)

	import numpy as np
	seqs[0] = seqs[0].lower()
	# seqs[0] = seqs[0].replace('n', '')
	a = list(seqs[0])
	x = a
	y = [ord(j) for j in x]
	del seqs
	del a
	del x
	print(y[0:30])

	out = np.array(y)
	del y
	let = np.unique(out)

	let = [chr(j) for j in let]
	print("Array contains", let)

	from sklearn.preprocessing import LabelEncoder
	from sklearn.preprocessing import OneHotEncoder
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(out)
	del out

	# print(integer_encoded)
	# onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	print(integer_encoded[0:10])
	# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	
	name = 'chromosomes_N/chr' + str(i) 
	print("Saving", name)
	np.save(name, integer_encoded)

