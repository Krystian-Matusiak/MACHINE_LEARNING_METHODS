# Needed libraries
from cProfile import label
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def plot_function(X,Y,x_label,y_label,title):
    plt.plot(X,Y,label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

def print_results(model,model_name,X,x_poly,Y,Y_pred):
    print(f"\n\n=============={model_name}==============")
    print(model.get_params())
    print("Score = ",model.score(x_poly,Y))
    print("Coefficients = ",model.coef_)
    print("Intercepts = ",model.intercept_)
    print(f"Equation = {model.coef_[0]}*x + {model.intercept_[0]}")
    print("Mean squared error = ",mean_squared_error(Y,Y_pred))


#   -------------------------------------------------------------------------------
if __name__ == "__main__":

    AMOUN = 500
    X = 0.1 * np.linspace(-10,10,AMOUN).reshape(AMOUN,1)
    Y = 3*X**3  + 0.5*X**2 + X + 2 + np.random.rand(AMOUN,1)

    poly = PolynomialFeatures(degree=3)
    x_poly = poly.fit_transform(X)

    lr = LinearRegression()
    lr.fit(x_poly,Y)
    y_lr = lr.predict(x_poly)

    print_results(lr,"LinearRegression",X,x_poly,Y,y_lr)

#   -------------------------------------------------------------------------------
    plot_function(X, Y,"X values","Y values","Exact values")
    plot_function(X, y_lr,"X values","Predicted values","Prediction for polynomial LINEAR regression")

    plt.legend()
    plt.show()
