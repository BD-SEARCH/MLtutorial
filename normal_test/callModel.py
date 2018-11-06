# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 1. data 준비
test_datagen = ImageDataGenerator(rescale=1./255)
# 검증용 generator 생성
test_generator = test_datagen.flow_from_directory(
        './hard_handwriting_shape/test',
        target_size=(24, 24),
        batch_size=3,
        class_mode='categorical')

# 2. call model
from keras.models import load_model
model = load_model('first_model.h5')

# 3. use model
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
print(test_generator.filenames)
