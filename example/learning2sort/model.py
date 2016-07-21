from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, TimeDistributedDense, Dropout, Dense
from keras.layers import recurrent
import numpy as np
from data import batch_gen
RNN = recurrent.LSTM

# global params
batch_size=32
seq_len = 10
max_no = 10

model = Sequential()
model.add(RNN(100, input_shape=(seq_len, max_no)))
model.add(Dropout(0.25))
model.add(RepeatVector(seq_len))
model.add(RNN(100, return_sequences=True))

model.add(TimeDistributedDense(max_no))
model.add(Dropout(0.5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

for X,Y in batch_gen(batch_size, seq_len, max_no):
    loss, acc = model.train_on_batch(X, Y)
    print loss, acc


