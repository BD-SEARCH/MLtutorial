from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = '/Users/soyoung/MLtutorial/img/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)

# # extract features
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print('Predicted:', decode_predictions(features, top=3)[0])
