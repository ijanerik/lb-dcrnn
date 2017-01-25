'''
The actual trainer!
'''
from __future__ import print_function
import numpy as np
from keras.models import Sequential
import keras.layers as layer
import lbf

np.random.seed(1337)  # for reproducibility

# CONFIG
nb_epoch = 2

# Load images
input_shape = lbf.get_input_shape()
images = lbf.load_images('images/')
norm_images = lbf.normalize_data(images, input_shape)
((X_train, Y_train), (X_test, Y_test)) = lbf.split_data(norm_images, 0.9)

# The actual model
model = Sequential()
model.add(layer.Convolution2D(16, 3, 3,
                        border_mode='valid',
                        input_shape=input_shape,
                        activation='relu'))
model.add(layer.Convolution2D(32, 2, 2, activation='relu'))
model.add(layer.Dropout(0.25))

model.add(layer.Convolution2D(32, 3, 3, activation='relu'))
model.add(layer.Convolution2D(64, 2, 2, activation='relu'))
model.add(layer.Dropout(0.25))

model.add(layer.Flatten())
model.add(layer.Dense(256, activation='relu'))
model.add(layer.Dropout(0.5))

# RNN
model.add(layer.RepeatVector(7))
model.add(layer.GRU(128, return_sequences=True, activation='relu'))
model.add(layer.TimeDistributed(layer.Dense(38, activation='softmax')))


# Compile and train!
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=256, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

# Save the model
lbf.save_model(model)
