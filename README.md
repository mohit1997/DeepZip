# DeepZip

<em>Update: Please checkout our new work [DZip](https://github.com/mohit1997/Dzip-torch) presented at DCC 2021.</em>

## Description
Data compression using neural networks

[DeepZip: Lossless Data Compression using Recurrent Neural Networks](https://arxiv.org/abs/1811.08162)

## Requirements
0. GPU, nvidia-docker (or try alternative installation)
1. python 2/3
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (cpu/gpu) 1.8

(nvidia-docker is currently required to run the code)
A simple way to install and run is to use the docker files provided:

```bash
cd docker
make bash BACKEND=tensorflow GPU=0 DATA=/path/to/data/
```

## Alternative Installation
```bash
cd DeepZip
python3 -m venv tf
source tf/bin/activate
bash install.sh
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
./run_experiments.sh biLSTM GPUID
```
Note: GPUID by default can be set to 0. The corresponding command would be then `./run_experiments.sh biLSTM 0`
### Please cite if you utilize the code in this repository.
```

@inproceedings{7fcb664b03ac4d6497048954d756b91f,
title = "DeepZip: Lossless Data Compression Using Recurrent Neural Networks",
author = "Mohit Goyal and Kedar Tatwawadi and Shubham Chandak and Idoia Ochoa",
year = "2019",
month = "5",
day = "10",
doi = "10.1109/DCC.2019.00087",
language = "English (US)",
series = "Data Compression Conference Proceedings",
publisher = "Institute of Electrical and Electronics Engineers Inc.",
editor = "Ali Bilgin and Storer, {James A.} and Marcellin, {Michael W.} and Joan Serra-Sagrista",
booktitle = "Proceedings - DCC 2019",
address = "United States",

}

```
