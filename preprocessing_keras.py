
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import codecs
import pdb
import numpy as np

class preprocess(object):
    # Preprocessing for hindi to english
    def __init__(self, path_hindi, path_eng, max_feat, max_len, truncate='post'):
        self.path_hindi = path_hindi
        self.path_eng = path_eng
        self.max_feat = max_feat
        self.max_len = max_len
        self.vocab_hind = []
        self.vocab_eng = []
        self.truncate = truncate

    def preprocess(self):
        # Preprocessing hindi corpus

        self.text_hindi =[line.rstrip() for line in open(self.path_hindi).readlines()]
        #self.text_eng = [line.rstrip() for line in codecs.open(self.path_eng, encoding='utf-8').readlines()]
        self.text_eng = [line.rstrip() for line in open(self.path_eng).readlines()]

        self.tokenizer = Tokenizer(self.max_feat)
        self.tokenizer.fit_on_texts(self.text_eng)
        self.vocab_en = self.tokenizer.word_index
        self.vocab_en["UNK"] = self.max_feat
        self.vocab_en["<s>"] = self.max_feat + 1
        self.vocab_en["</s>"] = self.max_feat + 2
        self.tokenizer1 = Tokenizer(self.max_feat)
        self.tokenizer1.fit_on_texts(self.text_hindi)
        self.vocab_hind = self.tokenizer1.word_index
        self.vocab_hind["UNK"] = self.max_feat
        self.vocab_hind["<s>"] = self.max_feat + 1
        self.vocab_hind["</s>"] = self.max_feat + 2
        text1 = self.tokenizer.texts_to_sequences(self.text_eng)
        text2 = self.tokenizer1.texts_to_sequences(self.text_hindi)

    def gen_seq(self, text_seq, text_seq1):
        nonzero_ind = []
        for ind, seq in enumerate(zip(text_seq, text_seq1)):
            if len(seq[0]) !=0 and len(seq[1]) !=0:
                nonzero_ind.append(ind)
        #nonzero_ind = [ind for ind, seq in enumerate(zip(text_seq, text_seq1)) if len(seq[0]) != 0 and len(seq[1]) != 0]

        text_seq_Y = [text_seq1[i] for i in nonzero_ind]
        text_seq_X = [text_seq[i] for i in nonzero_ind]
        # Normalize all sequences to maxlen
        #X = pad_sequences(text_seq_X, self.max_len)
        #Y = pad_sequences(text_seq_Y, self.max_len)
        X = np.zeros((len(text_seq_X), self.max_len + 2), np.int32)
        Y = np.zeros((len(text_seq_Y), self.max_len + 2, self.max_feat + 3), dtype=np.int32)
        #pdb.set_trace()
        for ind, seq in enumerate(zip(text_seq_X, text_seq_Y)):
            if len(seq[0]) <= (self.max_len):
                X[ind, 0] = self.vocab_en["<s>"]
                X[ind, 1:len(seq[0])+1] = seq[0]
                X[ind, len(seq[0])+1:] = self.vocab_en["</s>"]

            elif len(seq[0]) > self.max_len:
                if self.truncate == 'post':
                    temp = seq[0][:self.max_len]
                    X[ind, 0] = self.vocab_en["<s>"]
                    X[ind, 1:len(temp)+1] = temp
                    X[ind, len(temp)+1:] = self.vocab_en["</s>"]
                else:
                    temp = self[0][-self.max_len:]
                    X[ind, 0] = self.vocab_en["<s>"]
                    X[ind, 1:(len(temp) + 1)] = temp
                    X[ind, (len(temp) + 1):] = self.vocab_en["</s>"]

            #pdb.set_trace()
            if len(seq[1]) <= self.max_len:
                Y[ind, 0, self.vocab_en["<s>"]] = 1
                # 2nd dim ind + 1 because ind1 starts with 0 and 0 is filled
                # with <s>
                for ind1,j in enumerate(seq[1]):
                    Y[ind, (ind1 + 1), j] = 1

                for ind1 in xrange(len(seq[1])+1, self.max_len+2):
                    Y[ind, ind1, self.vocab_en["</s>"]] = 1

            elif len(seq[1]) > self.max_len:
                if self.truncate == 'post':
                    temp = seq[1][:self.max_len]

                else:
                    temp = seq[1][-self.max_len:]

                Y[ind, 0, self.vocab_en["<s>"]] = 1

                for ind1,j in enumerate(temp):
                    Y[ind, (ind1 + 1), j] = 1

                for ind1 in xrange(len(temp) + 1, self.max_len + 2):
                    Y[ind, ind1, self.vocab_en["</s>"]] = 1
        #pdb.set_trace()
        return X, Y


    def gen_batch(self, batch_size=32):
        i = 0
        text_seq = []
        text_seq1 = []
        for text1, text2 in zip(self.tokenizer.texts_to_sequences_generator(self.text_eng), self.tokenizer1.texts_to_sequences_generator(self.text_hindi)):
            if i == batch_size:
                X, Y = self.gen_seq(text_seq, text_seq1)
                text_seq = []
                text_seq1 = []
                i = 0
                yield X, Y
            text_seq.append(text1)
            text_seq1.append(text2)
            i += 1
                #pdb.set_trace()



if __name__ == "__main__":
    pre = preprocess('../indian-parallel-corpora/hi-en/tok/training.hi-en.hi', '../indian-parallel-corpora/hi-en/tok/training.hi-en.en', 5000, 50)
    pre.preprocess()
    for X,Y in pre.gen_batch():
        print X.shape, Y.shape
        pdb.set_trace()
