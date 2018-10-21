# DNA_Compression

## Description
DNA_compression using neural networks

Usage:
1. Copy the npy file required to this directory.
2. To run any of the code, use the command
```python
python script.py -d chr1.npy -gpu gpuid -name 'name of logs and weights file you want'
# Here gpuid can be 0 to n-1 GPUs you have
# chr1.npy can be replaced with any other sequence file, ex. markov_seq.npy
# -name argument will save the weights and the logs with the name you provide here
```

## Requirements
1. python 2/3
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (cpu/gpu) 1.8
6. argparse
