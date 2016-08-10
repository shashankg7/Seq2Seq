
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, TimeDistributedDense, Dropout, Dense
from keras.layers import recurrent
from keras.layers.embeddings import Embedding
import numpy as np
from preprocessing_keras import preprocess
import pdb
RNN = recurrent.LSTM

class seq2seq(object):
    # Definitions of various seq2seq models
    def __init__(self, input_size, seqlen, output_size, input_dim = 100, \
                 hidden_dim = 100):
        self.maxlen = seqlen
        self.input_size = input_size
        self.output_size = output_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def seq2seq_plain(self):

        model = Sequential()
        model.add(Embedding(self.input_size , self.input_dim))
        model.add(RNN(self.hidden_dim, return_sequences=True))#, input_shape=(100, 128)))

        model.add(Dropout(0.25))
        model.add(RNN(self.hidden_dim))
        model.add(RepeatVector(self.maxlen))
        model.add(RNN(self.hidden_dim, return_sequences=True))

        model.add(Dropout(0.25))
        model.add(RNN(self.hidden_dim, return_sequences=True))

        model.add(TimeDistributedDense(self.output_size))
        model.add(Dropout(0.5))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        return model

    def seq2seq_attention(self):
        raise NotImplementedError


if __name__ == "__main__":
    seq2seq = seq2seq(15, 5500)
    seq2seq.train_seq2seq()
