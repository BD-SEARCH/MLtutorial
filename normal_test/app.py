import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 랜덤시드 고정시키기
np.random.seed(3)

# 1. 데이터 생성
# train_datagen = ImageDataGenerator(rescale=1./255)

# 변화를 줘서 부풀리기.
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=15,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.8, 2.0],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')
# 훈련용 generator 생성
train_generator = train_datagen.flow_from_directory(
        './hard_handwriting_shape/train', # img 경로
        target_size=(24, 24), # 패치 이미지 크기
        batch_size=3, # 배치 크기
        class_mode='categorical') # categorical/binary/sparse/None

test_datagen = ImageDataGenerator(rescale=1./255)
# 검증용 generator 생성
test_generator = test_datagen.flow_from_directory(
        './hard_handwriting_shape/test',
        target_size=(24, 24),
        batch_size=3,
        class_mode='categorical')

# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit_generator(train_generator, steps_per_epoch=15, epochs=50, validation_data=test_generator, validation_steps=5)

# 5. 모델 평가하기
print("-- Evaluate(정확도) --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 6. 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
print(test_generator.filenames)
