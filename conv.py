''' Simple ConvNet for training in the first week!
A lot of redundant code, but makes it still working
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from PIL import Image
import glob
import math

batch_size = 256
nb_classes = 80
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 30, 60


# Find unique labels
def f3(seq):
   # Not order preserving
   keys = {}
   for e in seq:
       keys[e] = 1
   return keys.keys()

def readIn(path):
    # Read in labels
    images = []
    labels = []
    i = 0
    for img in glob.glob(path+'/*'):

        i = i + 1
        if(i % 10000 == 0):
            print("Loaded images: "+str(i))
        try:
            images.append(np.array(Image.open(img)))
            labels.append(img.split('.')[0].split('_')[1])

    uLabels = f3(labels)

    iLabels = []
    for label in labels:
        iLabels.append(uLabels.index(label))

    X = np.array(images)
    y = np.array(iLabels)

    trainSize = int(math.floor(len(X) * 0.85))
    permutation = np.random.permutation(y.shape[0])
    (per_train, per_test) = (permutation[: trainSize], permutation[trainSize :])
    (X_train, X_test) = (X[per_train], X[per_test])
    (y_train, y_test) = (y[per_train], y[per_test])
    uLabels = uLabels

    return (X_train, y_train, X_test, y_test, uLabels)




# the data, shuffled and split between train and test sets
(X_train, y_train, X_test, y_test, ul) = readIn('images')

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
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(16, 3, 3,
                        border_mode='valid',
                        input_shape=input_shape,
                        activation='relu'))
model.add(Convolution2D(16, 2, 2, activation='relu'))
# model.add(Convolution2D(16, 2, 2, activation='relu'))
#model.add(Convolution2D(16, 2, 2, activation='relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
print('Test score:', score[0])
print('Test accuracy:', score[1])
