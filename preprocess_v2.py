
import re
from collections import defaultdict
from collections import Counter
import pdb
import numpy as np


class WordTable(object):

    def __init__(self, hind, eng, maxlen):
        #hind.append("UNK")
        #eng.append("UNK")
        self.hind = set(hind)
        self.eng = set(eng)
        self.maxlen = maxlen
        self.hind_vocab = dict((w, i) for i, w in enumerate(self.hind))
        self.hind_vocab = dict(sorted(self.hind_vocab.iteritems(), key=lambda x:-x[1])[:maxlen])
        self.hind_vocab["UNK"] = max(self.hind_vocab.keys() + 1
        self.hind_vocab_rev = dict((i,w) for i, w in enumerate(self.hind))
        self.eng_vocab = dict((w,i) for i, w in enumerate(self.eng))
        self.eng_vocab = dict(sorted(self.eng_vocab.iteritems(), key=lambda x:-x[1])[:maxlen])
        self.eng_vocab["UNK"] = max(self.eng_vocab.keys() + 1 
        self.eng_vocab_rev = dict((i, w) for i, w in enumerate(self.eng))

    def encode_hind(self, C):
        X = np.zeros((maxlen, len(self.hind_vocab)))
        for i, c in enumerate(C):
            X[i, self.hind_vocab[c]] = 1
        return X

    def encode_eng(self, C):
        X = np.zeros((maxlen, len(self.eng_vocab)))
        for i, c in enumerate(C):
            X[i, self.eng_vocab[c]] = 1
        return X

    def decode_hind(self, X):
        X = X.argmax(axis=-1)
        return ''.join(self.hind_vocab_rev[ind] for ind in X)

    def decode_eng(self, X):
        X = X.argmax(axis=-1)
        return ''.join(self.eng_vocab_rev[ind] for ind in X)


class preprocess(object):

    def __init__(self, train_hi, train_eng, maxlen, maxfeat, batch_size):
        self.train_hi = train_hi
        self.train_eng = train_eng
        self.maxlen = maxlen
        self.maxfeat = maxfeat
        self.batch_size = batch_size 
        self.training_size = 50000
        self.hidden_size = 128
        
    def train(self):
        f = open(self.train_hi)
        f1 = open(self.train_eng)
        hind = ''.join((f.read().strip().lower().split("\n")))
        eng = ''.join((f.read().strip().lower().split("\n")))
        self.word_table = wordTable(hind, eng, self.maxlen)
        print "Vectorization"
        X = np.zeros(

if __name__ == "__main__":
    proproces = preprocess("../indian-parallel-corpora/hi-en/tok/training.hi-en.hi",
                           "../indian-parallel-corpora/hi-en/tok/training.hi-en.en", 128, 10000)
    proproces.gen_vocab()
    for (X,Y) in proproces.gen_seq("../indian-parallel-corpora/hi-en/tok/dev.hi-en.en.0",
                                 "../indian-parallel-corpora/hi-en/tok/dev.hi-en.hi",
                                   32):
        print len(Y)
        pdb.set_trace()


