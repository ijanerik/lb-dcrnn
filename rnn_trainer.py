'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GRU, RepeatVector
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from load_rnn import readIn
from encode_rnn import encodeHouseNumber, decodeHouseNumber

batch_size = 256
nb_epoch = 2
maxLen = 7

# input image dimensions
img_rows, img_cols = 30, 60

# the data, shuffled and split between train and test sets
(X_train, y_train, X_test, y_test) = readIn('images')

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = encodeHouseNumber(y_train, maxLen)
Y_test = encodeHouseNumber(y_test, maxLen)

model = Sequential()

model.add(Convolution2D(32, 3, 3,
                        border_mode='valid',
                        input_shape=input_shape,
                        activation='relu'))
model.add(Convolution2D(16, 2, 2, activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# RNN
model.add(RepeatVector(maxLen))
model.add(GRU(38, return_sequences=True))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

# Create name
import time
name = "model_"+time.strftime("%d-%m-%Y_%I:%M:%S")

# Save model
file = open("trained/"+name+".json", "w+")
file.write(model.to_json())
file.close()

# Save weights
model.save_weights('trained/'+name+'.h5')
