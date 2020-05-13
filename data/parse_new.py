import sys
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-param_file', action='store', dest='param_file',
                    help='param file file')
parser.add_argument('-input', action='store', dest='input_file_path',
                    help='input file path')
parser.add_argument('-output', action='store',dest='output_file_path',
                    help='output file path')
args = parser.parse_args()

with open(args.input_file_path, 'rb') as fp:
    data = fp.read()

print(len(data))
vals = list(set(data))
char2id_dict = {c: i for (i,c) in enumerate(vals)}
id2char_dict = {i: c for (i,c) in enumerate(vals)}

params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict}
with open(args.param_file, 'w') as f:
    json.dump(params, f, indent=4)

print(char2id_dict)
print(id2char_dict)

out = [char2id_dict[c] for c in data]
integer_encoded = np.array(out)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
print(integer_encoded[:10])
print(data[:10])

np.save(args.output_file_path, integer_encoded)
