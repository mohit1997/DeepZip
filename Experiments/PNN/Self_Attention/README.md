# DNA_Compression

## Parsing Fastq files and conversion to NPY(numpy) format
1. Download and extract all the zip genomic files in [Parsing/fastq_files/](Parsing/fastq_files/) with .fa format.
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
5. You will manually need to copy the files in the [Attention_Model](Attention_Model) folder

## Requirements
1. python 2/3
2. numpy
3. sklearn

Note: Already preprocessed file for chromosome 1 can be dowloaded from [here](https://www.dropbox.com/s/88ozf33cqsemzah/reference.npy?dl=0).