# DNA_Compression

## Parsing Fastq files and conversion to NPY(numpy) format
1. Download and extract all the zip genomic files in [fastq_files/](fastq_files/) with .fa format.
2. Edit the filename in the [parse.py](parse.py), the default format is chr1.fa for chromosome 1. If using the same scheme just edit the following list in the same file
```python
## set range to choose which chromosomes files to parse
## edit chromosome_list appropriately 
chromosomes_list = [1]
# This will generate only chr1.npy if chr1.fa is present
```
3. Run 
```python 
python parse.py
```
4. All numpy files are now created in [chromosomes_N](Parsing/chromosomes_N) folder.

## Requirements
1. python 2/3
2. numpy
3. sklearn

## Note: 
Already preprocessed file for chromosome 1 can be dowloaded from [here](https://www.dropbox.com/s/88ozf33cqsemzah/reference.npy?dl=0).