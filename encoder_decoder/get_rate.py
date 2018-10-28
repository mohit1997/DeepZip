import numpy as np

pb = np.load('prob_temp.npy')
lb = np.load('short.npy')

logits = np.log(pb[np.arange(len(lb)), lb[:, 0]] + 1e-8)

print(-np.mean(logits)/np.log(2))

