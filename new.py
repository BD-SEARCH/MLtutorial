# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

img_w, img_h = 150, 150

weight_name = 'first_try.h5'
train_data_dir = './dataset/training_set'
test_data_dir = './dataset/test_set'
nb_train_samples = 2000
nb_test_samples = 800
epochs = 50
batch_size = 16

def save_features():
    datagen = ImageDataGenerator(rescale=1./255)

    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    train_features = model.predict_generator(generator, nb_train_samples)
    np.save(open('train_features.npy','w'), train_features)

    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_w, img_h),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    test_features = model.predict_generator(generator, nb_test_samples)
    np.save(open('test_features.npy','w'), test_features)

def train_top_model():
    train_data = np.load(open('train_features.npy'))
    train_labels = np.array([0]*(nb_train_samples/2)+[1]*(nb_train_samples/2))

    test_data = np.load(open('test_features.npy'))
    test_labels = np.array([0]*(nb_test_samples/2)+[1]*(nb_test_samples/2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, test_data=(test_data, test_labels))
    model.save_weights(weight_name)

save_features()
train_top_model()

# # 1. data 준비
# test_datagen = ImageDataGenerator(rescale=1./255)
# # 검증용 generator 생성
# test_generator = test_datagen.flow_from_directory(
#         './dataset/test_set',
#         target_size=(24, 24),
#         batch_size=3,
#         class_mode='categorical')
#
# # 2. call model
# from keras.models import load_model
# model = load_model('first_model.h5')
#
# # 3. use model
# print("-- Predict --")
# output = model.predict_generator(test_generator, steps=5)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print(test_generator.class_indices)
# print(output)
# print(test_generator.filenames)
