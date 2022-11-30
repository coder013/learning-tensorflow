import os

import pandas as pd

dir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.abspath(os.path.join(dir, os.path.pardir)), "data")

lemonade = pd.read_csv(os.path.join(dataDir, "lemonade.csv"))
boston = pd.read_csv(os.path.join(dataDir, "boston.csv"))
iris = pd.read_csv(os.path.join(dataDir, "iris.csv"))

lemonadeIndependent = lemonade[['온도']]
lemonadeDependent = lemonade[['판매량']]

bostonIndependent = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
bostonDependent = boston[['medv']]

irisIndependent = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
irisDependent = iris[['품종']]

print()
print("lemonade의 독립 변수:", lemonadeIndependent.shape, end=", ")
print("lemonade의 종속 변수:", lemonadeDependent.shape)
print(lemonade.head())
print()
print("boston의 독립 변수:", bostonIndependent.shape, end=", ")
print("boston의 종속 변수:", bostonDependent.shape)
print(boston.head())
print()
print("iris의 독립 변수:", irisIndependent.shape, end=", ")
print("iris의 종속 변수:", irisDependent.shape)
print(iris.head())
print()