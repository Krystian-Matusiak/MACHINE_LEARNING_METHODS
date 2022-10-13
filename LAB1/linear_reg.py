# Needed libraries
from cProfile import label
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def plot_function(X,Y,x_label,y_label,title):
    plt.scatter(X,Y,label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

def print_results(model,model_name,X,Y,Y_pred):
    print(f"\n\n=============={model_name}==============")
    print(model.get_params())
    print("Score = ",model.score(X,Y))
    print("Coefficients = ",model.coef_)
    print("Intercepts = ",model.intercept_)
    print(f"Equation = {model.coef_[0]}*x + {model.intercept_[0]}")
    print("Mean absolute error = ",mean_absolute_error(Y,Y_pred))
    print("Mean squared error = ",mean_squared_error(Y,Y_pred))


#   -------------------------------------------------------------------------------
if __name__ == "__main__":

    plt.style.use('ggplot')

    AMOUN = 500
    X = 0.4 * np.linspace(-3,3,AMOUN).reshape(AMOUN,1)
    Y = 6 + 4*X + np.random.rand(AMOUN,1)

    lr = LinearRegression()
    sgdr = SGDRegressor()

    lr.fit(X, Y)
    sgdr.fit(X, Y)

    Y_lr_predict = lr.predict(X)
    Y_sgdr_predict = sgdr.predict(X)

    print_results(lr,"LinearRegression",X,Y,Y_lr_predict)
    print_results(sgdr,"SGDRegressor",X,Y,Y_sgdr_predict)

#   -------------------------------------------------------------------------------
    plot_function(X, Y,"X values","Y values","Exact values")
    plot_function(X, Y_lr_predict,"X values","Predicted values","Prediction for linear regression")
    plot_function(X, Y_sgdr_predict,"X values","Predicted values","Prediction for SGDRegressor")

    plt.legend()
    plt.show()

#   -------------------------------------------------------------------------------
    hyperparam = []
    MSE = []
    MAE = []
    for eps in np.arange(0.001,1,0.001):
        sgdr_ = SGDRegressor(eta0 = eps)
        sgdr_.fit(X,Y)
        MSE.append(mean_squared_error(Y,sgdr_.predict(X)))
        MAE.append(mean_absolute_error(Y,sgdr_.predict(X)))
        hyperparam.append(eps)
    plt.plot(hyperparam,MAE,label="MAE")    
    plt.plot(hyperparam,MSE,label="MSE")    
    plt.xlabel("eta0")
    plt.ylabel("Error")
    plt.title("Eta0 parameterization")
    plt.grid(True)
    plt.legend()
    plt.show()
