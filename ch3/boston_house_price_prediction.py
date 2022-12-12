import os

import pandas as pd
import tensorflow as tf

dir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.abspath(os.path.join(dir, os.path.pardir)), "data")

boston = pd.read_csv(os.path.join(dataDir, "boston.csv"))

bostonIndependent = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
bostonDependent = boston[['medv']]

X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)

model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.fit(bostonIndependent, bostonDependent, epochs=1000, verbose=0)
model.fit(bostonIndependent, bostonDependent, epochs=10)

print(bostonDependent[0:10])
print(model.predict(bostonIndependent[0:10]))

print(model.get_weights())