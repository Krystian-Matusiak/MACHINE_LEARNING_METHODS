# Needed libraries
import sklearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor


def plot_function(X,Y,x_label,y_label,title):
    plt.plot(X,Y,label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)


if __name__ == "__main__":
    X = 0.4 * np.linspace(-3,3,500).reshape(500,1)
    Y = 6 + 4*X + np.random.rand(500,1)

    lr = LinearRegression()
    sgdr = SGDRegressor()

    lr.fit(X, Y)
    sgdr.fit(X, Y)

    plot_function(X, Y,"X values","Y values","Exact values")
    plot_function(X, lr.predict(X),"X values","Predicted values","Prediction for linear regression")
    plot_function(X, sgdr.predict(X),"X values","Predicted values","Prediction for SGDRegressor")

    plt.legend()
    plt.show()
