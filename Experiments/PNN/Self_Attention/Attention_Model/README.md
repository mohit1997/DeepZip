# DNA_Compression

Implementation by [kaushalshetty](https://github.com/kaushalshetty/Structured-Self-Attention)
Paper Implemented [A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)

## How to run Attention Model on the DNA DATA
1. Copy the `.npy` file in the folder
2. Edit the filename in the [Parsing/parse.py](Parsing/parse.py), the default format is chr1.fa for chromosome 1. If using the same scheme just edit the following list in the same file
```python
## set range to choose which chromosomes files to parse
## edit chromosome_list appropriately
chromosomes_list = [1]
```
3. Run 
```python 
python parse.py
```
4. All numpy files are now created in [Parsing/chromosomes_N](Parsing/chromosomes_N) folder.


## Requirements
1. python 2/3
2. numpy
3. sklearn
