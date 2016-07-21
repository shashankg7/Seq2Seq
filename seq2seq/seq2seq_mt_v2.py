

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, TimeDistributedDense, Dropout, Dense
from keras.layers import recurrent
from keras.layers.embeddings import Embedding
import numpy as np
from preprocess_v2 import preprocess
import pdb
RNN = recurrent.LSTM

class seq2seq(object):
    # Train a sequence to sequence model for hindi to english translation
    def __init__(self, maxlen, batch):
        self.maxlen = maxlen
        self.batch = batch
        self.proproces = preprocess("../indian-parallel-corpora/hi-en/tok/training.hi-en.hi",
                           "../indian-parallel-corpora/hi-en/tok/training.hi-en.en", maxlen, 10000 )
        self.proproces.gen_vocab()
        #self.X_train = self.proproces.gen_seq_X("/home/shashank/datasets/MT-dataset/indian-parallel-corpora/hi-en/tok/training.hi-en.hi",
        #              "hindi")
        #self.Y_train = self.proproces.gen_seq_X("/home/shashank/datasets/MT-dataset/indian-parallel-corpora/hi-en/tok/training.hi-en.en",
        #              "english")
        #self.X_val = self.proproces.gen_seq_X("/home/shashank/datasets/MT-dataset/indian-parallel-corpora/hi-en/tok/dev.hi-en.hi",
        #              "hindi")
        #self.Y_val = self.proproces.gen_seq_X("/home/shashank/datasets/MT-dataset/indian-parallel-corpora/hi-en/tok/dev.hi-en.en.0",
        #              "english")
        #self.X_test = self.proproces.gen_seq_X("/home/shashank/datasets/MT-dataset/indian-parallel-corpora/hi-en/tok/test.hi-en.hi",
        #              "hindi")
        #self.Y_test = self.proproces.gen_seq_X("/home/shashank/datasets/MT-dataset/indian-parallel-corpora/hi-en/tok/test.hi-en.en.0",
        #              "english")

        #pdb.set_trace()
        #indices = np.arange(len(self.X_train))
        #np.random.shuffle(indices)
        #self.X_train = self.X_train[indices]
        #self.Y_train = self.Y_train[indices]

    def decode(self, prob):
        n_sent = prob.shape[0]
        for sent_id in range(n_sent):
            eng_id = prob[sent_id].argmax(axis=-1)
            print eng_id

    def train_seq2seq(self):
        print "Input sequence read, starting training"
        #X_train = sequence.pad_sequences(self.X_train, maxlen=self.maxlen)
        #Y_train = sequence.pad_sequences(self.Y_train, maxlen=self.maxlen)
        #X_val = sequence.pad_sequences(self.X_val, maxlen=self.maxlen)
        #y_val = sequence.pad_sequences(self.Y_val, maxlen=self.maxlen)
        #X_test = sequence.pad_sequences(self.X_test, maxlen=self.maxlen)
        #Y_test = sequence.pad_sequences(self.Y_test, maxlen=self.maxlen)

        model = Sequential()
        #model.add(Embedding(len(self.proproces.vocab_hind), 100,
        #                    input_length=self.maxlen))
        model.add(RNN(80, input_shape=(self.maxlen, len(self.proproces.vocab_hind))))
        model.add(RepeatVector(self.maxlen))
        model.add(RNN(80, return_sequences=True))
        model.add(TimeDistributedDense(len(self.proproces.vocab_en)))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        for e in range(1, 2000):
            print("epoch %d" % e)
            for (X,Y) in self.proproces.gen_seq("../indian-parallel-corpora/hi-en/tok/dev.hi-en.en.0",
                                 "../indian-parallel-corpora/hi-en/tok/dev.hi-en.hi",
                                           64):
                loss, acc = model.train_on_batch(X, Y)#, batch_size=64, nb_epoch=1)
                print("Loss is %f, accuracy is %f " % (loss, acc))
            # After one epoch test one sentence
            if e % 10 == 0:
                print("Enter sentence in hindi")
                inp = raw_input().decode("utf-8")
                tokens = inp.split()
                seq = []
                for token in tokens:
                    if token in self.proproces.vocab_hind:
                        seq.append(self.proproces.vocab_hind[token])
                    else:
                        token = "UNK"
                        seq.append(self.proproces.vocab_hind[token])
                #seq = map(lambda x:self.proproces.vocab_hind[x], tokens)
                # Normalize seq to maxlen
                X = []
                x = []
                temp = [0] * (self.maxlen)
                temp[0:len(seq)] = seq
                for ind in temp:
                    t = [0] * len(self.proproces.vocab_hind)
                    t[ind] = 1
                    x.append(t)
                X.append(x)
                X = np.asarray(X)
                print len(X)
                prob = model.predict(X)
                self.decode(prob)
                print("Probabilities are", prob)

if __name__ == "__main__":
    seq2seq = seq2seq(32, 128)
    seq2seq.train_seq2seq()
