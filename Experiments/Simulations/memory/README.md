# DNA_Compression

## LSTM usage
1. To generate markovian data, use generate_markov.py
2. short.npy and short_hg18.npy are genomic sequences belonging to human chormosome 1.
3. lstm.py uses a single-output LSTM for predicition
4. lstm_multiout.py uses a multi-output output LSTM for predicition.
5. bilstm.py uses a bidirectional single-output LSTM for predicition
6. lstm_multiout.py uses a bidirectional multi-output output LSTM for predicition.

Note: Change which data is used in the any of the above code before using.

## Requirements
1. python 2/3
2. keras 2.1.3
3. tensorflow (cpu/gpu) 1.8
