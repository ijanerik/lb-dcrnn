from keras.models import model_from_json
from keras import backend as K
from load_rnn import readIn
from encode_rnn import encodeHouseNumber, decodeHouseNumber

batch_size = 256
nb_epoch = 5
maxLen = 7

# input image dimensions
img_rows, img_cols = 30, 60

with open('/Users/janerik/Projecten/lb-dcrnn/trained/model_25-01-2017_01:41:28.json','r') as f:
    output = f.read()

model = model_from_json(output)
model.load_weights('/Users/janerik/Projecten/lb-dcrnn/trained/model_25-01-2017_01:41:28.h5')

### DATA READ:
# the data, shuffled and split between train and test sets
(X_train, y_train, X_test, y_test) = readIn('images')

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_train = X_train * 255

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# convert class vectors to binary class matrices
Y_train = encodeHouseNumber(y_train, maxLen)
total = 0
correct = 0
#print(model.evaluate(X_train, Y_train, batch_size=batch_size))
for i, val in enumerate(model.predict(X_train, batch_size=batch_size)):
    total = total + 1
    pred = decodeHouseNumber(Y_train[i], maxLen)
    out = decodeHouseNumber(val, maxLen)
    if(pred == out):
        correct = correct + 1
    print(pred, out)

acc = correct / float(total)
print("Accuracy:"+ str(acc))
