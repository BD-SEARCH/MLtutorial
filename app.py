#import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions


model = VGG16()
image = load_img('/Users/soyoung/MLtutorial/img/cat.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

yhat = model.predict(image)



print('Predicted:', decode_predictions(yhat, top=3)[0])

# convert the probabilities to class labels
#label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
#label = label[0][0]
# print the classification
#print('%s (%.2f%%)' % (label[1], label[2]*100))
