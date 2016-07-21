
import re
from collections import defaultdict
from collections import Counter
import pdb
import numpy as np
import codecs

class preprocess(object):

    def __init__(self, train_hi, train_eng, maxlen, maxfeat):
        self.train_hi = train_hi
        self.train_eng = train_eng
        self.maxlen = maxlen
        self.maxfeat = maxfeat
        self.vocab_en = defaultdict(int)
        self.vocab_en_rev = defaultdict()
        self.vocab_hind = defaultdict(int)
        self.count_en = defaultdict(int)
        self.count_hind = defaultdict(int)

    def gen_vocab(self):
        token_idx_hind = 0
        token_idx_en = 0
        f1 = codecs.open(self.train_hi, encoding='utf-8')
        for line in f1:
            tokens = line.rstrip().split()
            for token in tokens:
                #print token
                if token in self.vocab_hind:
                    self.count_hind[token] += 1

                elif token not in self.vocab_hind:
                    self.vocab_hind[token] = token_idx_hind
                    self.count_hind[token] += 1
                    token_idx_hind += 1

        self.vocab_hind["UNK"] = token_idx_hind
        f1.close()

        f1 = codecs.open(self.train_eng, encoding='utf-8')
        for line in f1:
            tokens = line.lower().rstrip().split()
            for token in tokens:
                if token in self.vocab_en:
                    self.count_en[token] += 1

                elif token not in self.vocab_en:
                    self.vocab_en[token] = token_idx_en
                    self.vocab_en_rev[token_idx_en] = token
                    self.count_en[token] += 1
                    token_idx_en += 1

        # Limit the vocab to top k most frequent unigrams
        unigram_hind = dict(Counter(self.count_hind).most_common(self.maxfeat)).keys()
        self.vocab_hind = dict(zip(unigram_hind, range(len(unigram_hind))))
        self.vocab_hind["UNK"] = len(self.vocab_hind)

        unigram_eng = dict(Counter(self.count_en).most_common(self.maxfeat)).keys()
        self.vocab_en = dict(zip(unigram_eng, range(len(unigram_eng))))
        self.vocab_en["UNK"] = len(self.vocab_en)
        self.vocab_en_rev[len(self.vocab_en)] = "UNK"
        #pdb.set_trace()
        #print self.vocab_en
        #print self.vocab_hind
        f1.close()

    def gen_seq(self, path_eng, path_hind, batch_size):
        X = []
        Y = []
        print "inside gen seq"
        f1 = codecs.open(path_eng, encoding='utf-8')
        f2 = codecs.open(path_hind, encoding='utf-8')
        i = 0
        print "insiede gen_seq"
        for line_en, line_hind in zip(f1, f2):
            x = []
            x1 = []
            y = []
            tokens_hind = line_hind.rstrip().split()
            for token in tokens_hind:
               # print token
                if token in self.vocab_hind:
                    x.append(self.vocab_hind[token])
                else:
                    token = "UNK"
                    x.append(self.vocab_hind[token])
            # Normalizing all sequences to "maxlen"
            if len(x) < self.maxlen:
                x2 = [0] * self.maxlen
                x2[0:len(x)] = x
                x = x2
            elif len(x) > self.maxlen:
                x = x[0:self.maxlen]
            else:
                pass
            X.append(x)
            # parse hindi sentences
            tokens_en = line_en.strip().split()
            for token in tokens_en:
                if token in self.vocab_en:
                    x1.append(self.vocab_en[token])
                else:
                    token = "UNK"
                    x1.append(self.vocab_en[token])
            if len(x1) < self.maxlen:
                x2 = [0] * self.maxlen
                x2[0:len(x1)] = x1
                x1 = x2
            elif len(x1) > self.maxlen:
                x1 = x1[0:self.maxlen]
            else:
                pass
            for ind in x1:
                temp = [0] * len(self.vocab_en)
                temp[ind] = 1
                y.append(temp)
            Y.append(y)
            i += 1
            if i % batch_size == 0:
                yield (X,Y)
                Y = []
                X = []
        #return np.asarray(X)
        print "in gen seq"
        pdb.set_trace()
        f1.close()
        f2.close()

    def gen_seq_Y(self, path, lang, batch_size):
        print "Inside func"
        X = []
        print "In gen seq Y"
        f = open(path)
        if lang == "hindi":
            vocab = self.vocab_hind
        else:
            vocab = self.vocab_en
        i = 0
        for line in f:
            x = []
            y = []
            tokens = line.strip().split()
            for token in tokens:
                if token in vocab:
                    x.append(vocab[token])
                else:
                    token = "UNK"
                    x.append(vocab[token])
            # Normalizing length of x to "maxlen"
            if len(x) < self.maxlen:
                x1 = [0] * self.maxlen
                x1[0:len(x)] = x
                x = x1
            elif len(x) > self.maxlen:
                x = x[0:self.maxlen]
            else:
                pass
            for ind in x:
                temp = [0] * len(vocab)
                temp[ind] = 1
                y.append(temp)

            X.append(y)
            i += 1
            if i % batch_size == 0:
                yield X
                X = []

if __name__ == "__main__":
    proproces = preprocess("../indian-parallel-corpora/hi-en/tok/training.hi-en.hi",
                           "../indian-parallel-corpora/hi-en/tok/training.hi-en.en", 128, 10000)
    print "Generating vocab"
    proproces.gen_vocab()
    #proproces.gen_seq("../indian-parallel-corpora/hi-en/tok/dev.hi-en.en.0",
    #                             "../indian-parallel-corpora/hi-en/tok/dev.hi-en.hi",
    #                               32)
    print "VOcav generated"
    proproces.gen_seq_Y("../indian-parallel-corpora/hi-en/tok/dev.hi-en.en.0",
                                 "hindi",
                                   32)

    print "generating sequence"
 #   pdb.set_trace()

