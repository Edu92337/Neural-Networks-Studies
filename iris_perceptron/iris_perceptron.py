from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
x = iris.data[:,(2,3)]
y = iris.target
perceptron = Perceptron()
history = perceptron.fit(x,y)
y_pred = perceptron.predict([[2,0.5]])
