from PIL import Image
import glob
import math
import numpy as np

# Find unique labels
def f3(seq):
   # Not order preserving
   keys = {}
   for e in seq:
       keys[e] = 1
   return keys.keys()

def scale256(X):
    return (X / 255. -.5) * 3.

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
