# DNA_Compression

## This code uses gradient clipping to allow learning long term dependencies.
Note: The LSTM used is of **multi-input single-output type** where each timestep predicts the next character.
```python
optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, clipnorm=0.05)
# clipnorm controls the gradient clipping by the optimizer
```
## How to use
1. Open [generate_markov.py](generate_markov.py) and change `np.logical_xor(l[-'the location of other bit'], l[-1])` without quotes.
2. Run [generate_markov.py](generate_markov.py)
3. Now run either [lstm_onehot.py](lstm_onehot.py) with one hot input encoding or [lstm_embeddings.py](lstm_embeddings.py) to use embeddings for input sequence.
## Requirements
1. python 2/3
2. numpy
3. sklearn
4. keras 2.1.3
5. tensorflow (cpu/gpu) 1.8
