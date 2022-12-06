import os

import pandas as pd
import tensorflow as tf

dir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.abspath(os.path.join(dir, os.path.pardir)), "data")

lemonade = pd.read_csv(os.path.join(dataDir, "lemonade.csv"))

lemonadeIndependent = lemonade[['온도']]
lemonadeDependent = lemonade[['판매량']]

X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)

model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.fit(lemonadeIndependent, lemonadeDependent, epochs=10000, verbose=0)
model.fit(lemonadeIndependent, lemonadeDependent, epochs=10)

print(model.predict(lemonadeIndependent))
print(model.predict([[26], [27], [28], [29], [30]]))