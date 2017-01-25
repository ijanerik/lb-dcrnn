import lbf

# input image dimensions
img_rows, img_cols = 30, 60

# Load images
input_shape = lbf.get_input_shape(img_cols, img_rows)
images = lbf.load_images('images/')
(X, Y) = lbf.normalize_data(images, input_shape)

# Load model
model = lbf.load_model('trained/model_25-01-2017_08:22:14')
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Do some nice retrieval stuff!!
total = 0
correct = 0
for i, val in enumerate(model.predict(X, batch_size=256)):
    total = total + 1
    [pred] = lbf.decodeHouseNumbers([Y[i]])
    [out] = lbf.decodeHouseNumbers([val])
    if(pred == out):
        correct = correct + 1
    print(pred, out)

acc = correct / float(total)
print("Accuracy:"+ str(acc))
