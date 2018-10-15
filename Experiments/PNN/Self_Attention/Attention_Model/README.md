# DNA_Compression

Implementation by [kaushalshetty](https://github.com/kaushalshetty/Structured-Self-Attention).
Paper Implemented [A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130).

## How to run Attention Model on the DNA DATA
1. Copy the `.npy` file in the folder
2. Edit the [utils/data_loader.py](utils/data_loader.py), in line no 133, and replace the name of the file:
```python
X_train, Y_train = generate_positive(300000000, file='name_of_numpy_file.npy', length=max_len, stride=1, full=True)
```

3. Run 
```python 
python classification.py 'dna'
```


## Requirements
1. python 2/3
2. numpy
3. sklearn
4. pytorch 0.4.1