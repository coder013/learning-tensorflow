import os

import pandas as pd
import tensorflow as tf

dir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.abspath(os.path.join(dir, os.path.pardir)), "data")

iris = pd.read_csv(os.path.join(dataDir, "iris.csv"))
encodedIris = pd.get_dummies(iris)

irisIndependent = encodedIris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
irisDependent = encodedIris[['품종_setosa', '품종_versicolor', '품종_virginica']]

X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(irisIndependent, irisDependent, epochs=100, verbose=0)
model.fit(irisIndependent, irisDependent, epochs=10)

print(irisDependent[0:5])
print(model.predict(irisIndependent[0:5]))

print(irisDependent[-5:])
print(model.predict(irisIndependent[-5:]))

print(model.get_weights())