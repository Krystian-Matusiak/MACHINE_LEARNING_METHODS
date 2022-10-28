# Needed libraries
from cProfile import label
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

def plot_function(X,Y_train,x_label,y_label,title):
    plt.scatter(X,Y_train,label=title, s=15)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

def print_results(model,model_name,X_test,Y_exact,Y_pred):
    print(f"\n\n=============={model_name}==============")
    print(model.get_params())
    print("Score = ",model.score(X_test,Y_exact))
    print("r2_score = ",r2_score(Y_exact,Y_pred))
    print("Coefficients = ",model.coef_)
    print("Intercepts = ",model.intercept_)
    print(f"Equation = {model.coef_[0]}*x + {model.intercept_[0]}")
    print("Mean absolute error = ",mean_absolute_error(Y_exact,Y_pred))
    print("Mean squared error = ",mean_squared_error(Y_exact,Y_pred))


#   -------------------------------------------------------------------------------
if __name__ == "__main__":

    plt.style.use('ggplot')
    AMOUN = 1000
    X = 0.4 * np.linspace(-3,3,AMOUN).reshape(AMOUN,1)
    X_train, X_test = train_test_split(X,train_size=0.5)
    # Not sure if there is a need to have separated x_train and x_test
    # X_train = X_test

    Y_train = 6 + 4*X_train + np.random.randn(int(AMOUN/2),1)
    Y_exact = 6 + 4*X_test + np.random.randn(int(AMOUN/2),1)
    # Y_exact = 6 + 4*X_train

    lr = LinearRegression()
    sgdr = SGDRegressor()

    # LinearRegression() time measurement
    lr_start = time.time()
    lr.fit(X_train, Y_train)
    lr_end = time.time()
    lr_fit_duration = lr_end - lr_start

    # SGDRegressor() time measurement
    sgd_start = time.time()
    sgdr.fit(X_train, Y_train)
    sgd_end = time.time()
    sgd_fit_duration = sgd_end - sgd_start

    print(f"Fit time duration for LR: {lr_fit_duration}")
    print(f"Fit time duration for SGD: {sgd_fit_duration}")

    Y_lr_predict = lr.predict(X_test)
    Y_sgdr_predict = sgdr.predict(X_test)

    print_results(lr,"LinearRegression",X_test,Y_exact,Y_lr_predict)
    print_results(sgdr,"SGDRegressor",X_test,Y_exact,Y_sgdr_predict)

#   -------------------------------------------------------------------------------
#   Plot all functions 
    plot_function(X_train, Y_train,"X values","Y_train values","Exact values")
    plot_function(X_test, Y_lr_predict,"X values","Predicted values","Prediction for linear regression")
    plot_function(X_test, Y_sgdr_predict,"X values","Predicted values","Prediction for SGDRegressor")
    plt.title("Plot for linear and SGD regression")

    plt.legend()
    plt.show()

#   -------------------------------------------------------------------------------
#   Plot eta parametrization
    hyperparam_eta = []
    Score = []
    for eta in np.arange(0.001,1,0.001):
        sgdr_ = SGDRegressor(eta0 = eta)
        sgdr_.fit(X_train,Y_train)
        Score.append(sgdr_.score(X_test,Y_exact))
        hyperparam_eta.append(eta)
    plt.plot(hyperparam_eta,Score)    
    plt.xlabel("eta0")
    plt.ylabel("Model score")
    plt.title("Eta0 parameterization")
    plt.grid(True)
    plt.legend()
    plt.show()

#   -------------------------------------------------------------------------------
#   Plot power_t parametrization
    hyperparam_power = []
    Score = []
    for power in np.arange(0.001,1,0.001):
        sgdr_ = SGDRegressor(eta0=0.01, power_t=power)
        sgdr_.fit(X_train,Y_train)
        Score.append(sgdr_.score(X_test,Y_exact))
        hyperparam_power.append(power)
    plt.plot(hyperparam_power,Score)    
    plt.xlabel("power_t")
    plt.ylabel("Model score")
    plt.title("power_t parameterization")
    plt.grid(True)
    plt.legend()
    plt.show()

#   -------------------------------------------------------------------------------
#   Plot Time(samples) and Score(samples)
    N = range(100,10000,50)
    Time_LR = []
    Time_SGD = []
    Score_LR = []
    Score_SGD = []
    for n in N:
        X_train = 0.4 * np.linspace(-3,3,n).reshape(n,1)
        Y_train = 6 + 4*X_train + np.random.randn(n,1)
        Y_exact = 6 + 4*X_train
        lr = LinearRegression()
        sgdr = SGDRegressor()

        # LinearRegression() time measurement
        lr_start = time.time()
        lr.fit(X_train, Y_train)
        lr_end = time.time()
        lr_fit_duration = lr_end - lr_start

        # SGDRegressor() time measurement
        sgd_start = time.time()
        sgdr.fit(X_train, Y_train)
        sgd_end = time.time()
        sgd_fit_duration = sgd_end - sgd_start

        Time_LR.append(lr_fit_duration)
        Time_SGD.append(sgd_fit_duration)
        Score_LR.append(r2_score(Y_exact,lr.predict(X_train)))
        Score_SGD.append(r2_score(Y_exact,sgdr.predict(X_train)))

    plt.plot(N,Time_LR,label="Time for LR")    
    plt.plot(N,Time_SGD,label="Time for SGD")    
    plt.xlabel("N samples")
    plt.ylabel("Time")
    plt.title("Relation between time and number of samples")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(N,Score_LR,label="Score for LR")    
    plt.plot(N,Score_SGD,label="Score for SGD")    
    plt.xlabel("N samples")
    plt.ylabel("Score")
    plt.title("Relation between score and number of samples")
    plt.grid(True)
    plt.legend()
    plt.show()