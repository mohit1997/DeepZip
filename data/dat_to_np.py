import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-infile', action='store', dest='infile', type = str, required=True)
parser.add_argument('-outfile', action='store', dest='outfile', type = str, required=True)
args = parser.parse_args()

with open(args.infile,'r') as f:
    l = [float(line.rstrip('\n')) for line in f.readlines()]

l_np = np.array(l, dtype=np.float32)
np.save(args.outfile, l_np)
