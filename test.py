from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train_std, Y_train, epochs=10, batch_size=105)
loss_and_metrics = model.evaluate(X_test_std, Y_test, batch_size=128)

classes = model.predict(X_test_std, batch_size=105)
