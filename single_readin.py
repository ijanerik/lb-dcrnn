import lbf

# input image dimensions
img_rows, img_cols = 30, 60

# Load images
input_shape = lbf.get_input_shape(img_cols, img_rows)
images = lbf.load_images('images/')
(X, Y) = lbf.normalize_data(images, input_shape)

# Load model
model = lbf.load_model('trained/model_25-01-2017_08:11:46')
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

for i, val in enumerate(model.predict(X, batch_size=256)):
    [pred] = lbf.decodeHouseNumbers([Y[i]])
    [out] = lbf.decodeHouseNumbers([val])
    print(pred, out)
