import numpy as np
import os
import io
import json
from nltk.tokenize import TweetTokenizer
from collections import defaultdict, Counter, OrderedDict

raw_data_path = 'save/ptb.train.txt'
data_dir = 'save'
vocab_file = 'train_vocab'
min_occ = 1

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)



def _create_vocab():
        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<GO>', '<PAD>', '<EOS>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(raw_data_path, 'r') as file:

            for i, line in enumerate(file):
                words = tokenizer.tokenize(line)
                w2c.update(words)

            for w, c in w2c.items():
                if c > min_occ:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))
        print(w2i)

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(data_dir, vocab_file), 'wb') as vo_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vo_file.write(data.encode('utf8', 'replace'))


_create_vocab()