# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam


# Definir constantes
dir_right = "C:/Users/Eliana/Desktop/Assessment 3 files/Right"
dir_left = "C:/Users/Eliana/Desktop/Assessment 3 files/Left"
dir_forward = "C:/Users/Eliana/Desktop/Assessment 3 files/Forward"
width = 40
height = 45
channels = 3
input_shape = (height, width, channels)

# Cargar los datos de entrenamiento
def load_data():
    images = []
    labels = []
    for dir_name, label in [(dir_right, 0), (dir_left, 1), (dir_forward, 2)]:
        for img_name in os.listdir(dir_name):
            img = cv2.imread(os.path.join(dir_name, img_name))
            img = cv2.resize(img, (width, height))
            images.append(img)
            labels.append(label)
    labels = np.eye(3)[labels]
    return np.array(images), np.array(labels)

images, labels = load_data()

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definir la arquitectura del modelo
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# Evaluar el modelo
score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])

#Guardar el modelo
model.save('model.h5')
