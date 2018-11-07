# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
#
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# include_top : not loaded the last 2 fully connected layers (classifier)
# 7 * 7 * 512
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# extract features
train_dir = '../dataset/training_set'
validation_dir = '../dataset/test_set'

nTrain = 100
nVal = 50
epochs=10

# load the images
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,3))

# generate batches of images and labels
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=None)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical'
)


i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nImages:
        break

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))


# make new model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

# train the model
model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

model.fit_generator(train_generator, steps_per_epoch=nTrain, epochs=epochs, validation_data=val_generator, validation_steps=nVal)
model.save_weights('vgg_model.h5')


# history = model.fit(train_features,
#                     train_labels,
#                     epochs=20,
#                     batch_size=batch_size,
#                     validation_data=(validation_features,validation_labels))


