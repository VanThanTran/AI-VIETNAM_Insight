import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# data preparation
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# normalize
x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_test.mean()) / x_test.std()

# model
model = keras.models.Sequential()
model.add(keras.Input(shape=(32, 32, 3)))

blocks = 5
convs = [2, 2, 3, 3, 3]
filters = [64, 128, 256, 512, 1024]
drops = [0.2, 0.3, 0.4, 0.5, 0.6]
initializer = keras.initializers.he_normal()

for block in range(blocks):
    for conv in range(convs[block]):
        model.add(keras.layers.Conv2D(filters[block], (3, 3), strides=1, padding='same', activation='relu',
                                      kernel_initializer=initializer))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Dropout(drops[block]))

# flatten
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256, activation='relu', kernel_initializer=initializer))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.7))

model.add(keras.layers.Dense(100, activation='softmax'))

# Declare optimization method and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
history_data = model.fit(x_train, y_train, batch_size=512,
                         validation_data=(x_test, y_test),
                         epochs=50, verbose=1)

# Graph
plt.plot(history_data.history['accuracy'], label="train_accuracy")
plt.plot(history_data.history['val_accuracy'], label="val_accuracy")
plt.xlabel('iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.show()