# 80%

# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

img_w, img_h = 150, 150
train_data_dir = './dataset/training_set'
test_data_dir = './dataset/test_set'
nb_train_samples = 100
nb_test_samples = 50
epochs = 25
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_w, img_h)
else:
    input_shape = (img_w, img_h, 3)

# 1. dataset 생성
# 변화를 줘서 부풀리기.
train_datagen = ImageDataGenerator(rescale=1./255,
                                #   rotation_range=15,
                                #   width_shift_range=0.1,
                                #   height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.8, 2.0],
                                  horizontal_flip=True,
                                #   vertical_flip=True,
                                  fill_mode='nearest')

# 훈련용 generator 생성
train_generator = train_datagen.flow_from_directory(
        train_data_dir, # img 경로
        target_size=(img_w, img_h), # 패치 이미지 크기
        batch_size=batch_size, # 배치 크기
        class_mode='categorical') # categorical/binary/sparse/None

test_datagen = ImageDataGenerator(rescale=1./255)
# 검증용 generator 생성
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        class_mode='categorical')



# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
# model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])





# 4. 모델 학습시키기
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples, epochs=epochs, validation_data=test_generator, validation_steps=nb_test_samples)
model.save_weights('test_model.h5')

# 5. 모델 평가하기
print("-- Evaluate(정확도) --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
#
# # 6. 모델 저장하기
# from keras.models import load_model
# model.save('testModel.h5')
