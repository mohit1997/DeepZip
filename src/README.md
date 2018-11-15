# DNA_Compression

## Description
DNA_compression using neural networks

This folder contains the final models. 

Important Notes:
1.
```python
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
# set save_weights_only argument according to your convenince
```
2. Run `python script.py -h` for help

## Requirements
1. python 2/3
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (cpu/gpu) 1.8
