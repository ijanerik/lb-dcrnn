from __future__ import print_function
from keras.models import model_from_json
from keras import backend as K
from PIL import Image
import numpy as np
import math, glob, time, random

import lbfg

########################################
## All the data for importing images! ##
########################################

def f3(seq):
   # Not order preserving
   keys = {}
   for e in seq:
       keys[e] = 1
   return list(keys.keys())

def labelToInt(labels):
    uLabels = f3(labels)
    iLabels = []
    for label in labels:
      iLabels.append(uLabels.index(label))
    return iLabels

def get_input_shape(cols = 60, rows = 30):
    if K.image_dim_ordering() == 'th':
        input_shape = (1, rows, cols)
    else:
        input_shape = (rows, cols, 1)
    return input_shape

def load_images(path):
    # Read in labels
    images = []
    labels = []
    i = 0
    for img in glob.glob(path+'/*'):
        i = i + 1
        if(i % 10000 == 0):
            print("Loaded images: "+str(i))

        images.append(np.array(Image.open(img)))
        labels.append(img.split('.')[0].split('_')[1])

    X = np.array(images)
    Y = np.array(labels)
    return (X, Y)

def normalize_data((X, Y), format, rnn=True, maxLen = 7, normalize = True):
    # Randomize input
    permutation = np.random.permutation(Y.shape[0])
    X = X[permutation]
    Y = Y[permutation]
    # Normalize
    if(normalize == True):
        X = (X / 255. - 0.5) * 6.

    # Normalize
    X = X.reshape(X.shape[0], format[0], format[1], format[2])
    X = X.astype('float32')

    # Encode Y
    if rnn == True:
        Y = encodeHouseNumbers(Y, maxLen)

    return (X, Y)

def split_data((X, Y), percentage = 0.6):
    trainSize = int(math.floor(len(X) * percentage))
    validationTest = len(X) - trainSize
    (X_train, X_valTest) = (X[: trainSize], X[trainSize: ])
    (Y_train, Y_valTest) = (Y[: trainSize], Y[trainSize: ])
    (X_validate, X_test) = (X_valTest[: X_valTest.shape[0]/2], X_valTest[X_valTest.shape[0]/2: ])
    (Y_validate, Y_test) = (Y_valTest[: Y_valTest.shape[0]/2], Y_valTest[Y_valTest.shape[0]/2: ])
    return [(X_train, Y_train), (X_validate, Y_validate), (X_test, Y_test)]

###########################################
## All the functions for generating data ##
###########################################
def generate_data(adresses, times, path = False, size = (60,30)):
    return lbfg.generate_data(adresses, times, path, size)



############################################################
## All the functions for encoding and decoding the labels ##
############################################################
def encodeHouseNumbers(data, maxLen = 7):
    '''
    Encode all the house numbers from a string into a matrix of categories
    '''
    chars = list(map(str, range(10))) + list(map(chr, range(65, 91))) + ['-', ' ']
    ret = np.zeros((len(data), maxLen, len(chars)))

    # Loop over all housenumbers to encode
    for i, text in enumerate(data):
        diffLength = maxLen - len(text)

        # Add spaces
        for o in range(diffLength):
            ret[i][o][len(chars) - 1] = 1

        # Over each character
        for o, char in enumerate(text):
            ret[i][diffLength+o][chars.index(char)] = 1
    return ret


def decodeHouseNumbers(data, maxLen = 7):
    '''
    Decode the matrix of the housenumbers back into a real house number!
    @TODO Make better with predicting confidence
    '''
    chars = list(map(str, range(10))) + list(map(chr, range(65, 91))) + ['-', ' ']
    ret = []
    for matrix in data:
        stri = []
        for i in matrix:
            stri.append(chars[np.argmax(i)])
        ret.append(''.join(stri))
    return ret


#############################################
## Save and load the model into/from files ##
#############################################
def save_model(model, path = 'trained/', prefix = "model_", optDrop=''):
    name = prefix+time.strftime("%d-%m-%Y_%I:%M:%S")

    # Save model
    file = open(optDrop+path+name+".json", "w+")
    file.write(model.to_json())
    file.close()

    # Save weights
    model.save_weights(path+name+'.h5')
    print('Saved the model in: '+ str(path+name))


def load_model(path):
    # Get json with model
    with open(path+'.json','r') as f:
        output = f.read()

    # Load model from json and load weights
    model = model_from_json(output)
    model.load_weights(path+'.h5')
    return model
