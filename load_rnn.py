from PIL import Image
import glob
import math
import numpy as np

def readIn(path, trainPercentage = 0.9):
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
    y = np.array(labels)

    trainSize = int(math.floor(len(X) * trainPercentage))
    permutation = np.random.permutation(y.shape[0])
    (per_train, per_test) = (permutation[: trainSize], permutation[trainSize :])
    (X_train, X_test) = (X[per_train], X[per_test])
    (y_train, y_test) = (y[per_train], y[per_test])

    return (X_train, y_train, X_test, y_test)
