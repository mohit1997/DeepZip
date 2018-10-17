# DNA_Compression

## This code uses gradeint clipping to allow learning long term dependencies.
```python
optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.05)
# clipnorm controls the gradient clipping by the optimizer
```
## Requirements
1. python 2/3
2. numpy
3. sklearn
4. keras 2.1.3
5. tensorflow (cpu/gpu) 1.8
