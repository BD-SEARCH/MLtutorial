# 0. call back the package
# import tensorflow as tf
# from tensorflow.keras.utils import np_utils
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# 1. ready datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. create datasets
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# 3. compose model
model = Sequential()
model.add(Dense(64, input_dim=28*28, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 4. set the model train set
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 5. train the model
fitted = model.fit(X_train, y_train, epochs=5, batch_size=32)

# 6. print
print('\n## training loss and accuracy ##')
print(fitted.history['loss'])
print(fitted.history['acc'])

# 7. evaluate
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
print("\n## evaluation loss and metrics ##")
print(loss_and_metrics)

# 8. start
Xhat = X_test[0:1]
yhat = model.predict(Xhat)
print('## yhat ##')
print(yhat)
