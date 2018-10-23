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

3. Alternately you can edit [Params_list.sh](Params_list.sh) and then run
```bash
bash Params_list.sh
```
4. Stateful Models are in [stateful](stateful).

Note: Please edit the flags appropriately to choose data in [Params_list.sh](Params_list.sh), one example is given in [Params_backup.sh](Params_backup.sh) to choose `chr1.npy`(chromosome 1)

## Requirements
1. python 2/3
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (cpu/gpu) 1.8
6. argparse
