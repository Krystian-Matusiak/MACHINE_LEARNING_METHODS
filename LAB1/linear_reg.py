# Needed libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def plot_function(X,Y,x_label,y_label,title):
    plt.plot(X,Y,label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

def print_results(model,model_name,X,Y,Y_pred):
    print(f"\n\n=============={model_name}==============")
    print(model.get_params())
    print("Model score = ",model.score(X,Y))
    print("Model coefficients = ",model.coef_)
    print("mean absolute error = ",mean_absolute_error(Y,Y_pred))
    print("mean squared error = ",mean_squared_error(Y,Y_pred))


if __name__ == "__main__":
    X = 0.4 * np.linspace(-3,3,500).reshape(500,1)
    Y = 6 + 4*X + np.random.rand(500,1)

    lr = LinearRegression()
    sgdr = SGDRegressor()

    lr.fit(X, Y)
    sgdr.fit(X, Y)

    Y_lr_predict = lr.predict(X)
    Y_sgdr_predict = sgdr.predict(X)

    plot_function(X, Y,"X values","Y values","Exact values")
    plot_function(X, Y_lr_predict,"X values","Predicted values","Prediction for linear regression")
    plot_function(X, Y_sgdr_predict,"X values","Predicted values","Prediction for SGDRegressor")

    print_results(lr,"LinearRegression",X,Y,Y_lr_predict)
    print_results(sgdr,"SGDRegressor",X,Y,Y_sgdr_predict)

    plt.legend()
    plt.show()
