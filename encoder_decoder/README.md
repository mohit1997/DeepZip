# DNA_Compression

## Description
The arithmetic encoder decoder for Neural Network Compressor
The model uploaded achieves 1.545 bits per base pair for the complete chromosome 1

## Usage
1. Open validate.py to compress `chr1.npy` generate `prob_temp.npy` file. You can change the name in the validate.py file for the chromosome to be compressed. It loads the model using `model.h5` file.
2. After the prob_temp.npy file is generated. Run get_rate.py to evaluate the compression rate.
3. Run `python dna_compress.py encoded.txt` to enocde the sequence in `out.txt`
4. Run `python dna_dynamic_decompress.py encoded.txt decoded.txt` to perform decompression.

## Requirements
1. python 2/3
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (cpu/gpu) 1.8
