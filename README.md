# DeepZip

## Description
DNA_compression using neural networks

This branch is the non-GPU implementation of deepzip.
*WARNING* The training is significantly slower on a CPU and the code will take significant longer than reporterd in the manuscript. 
The code has been tested on macOS Mojave. Please file an issue if there are problems.

## Requirements
1. python 2/3
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (cpu/gpu) 1.8

A simple way to install and run is to use the docker files provided:

```bash
cd docker
make bash BACKEND=tensorflow DATA=/path/to/data/
```

## Code
To run a compression experiment: 

### Data Preparation
1. Place all the data to be compressed in data/files_to_be_compressed
2. Run the parser 

```bash
cd data
./run_parser.sh
```

### Running models
1. All the models are listed in models.py
2. Pick a model, to run compression experiment on all the data files in the data/files_to_be_compressed directory

```
cd src
./run_experiments.sh biLSTM
```
