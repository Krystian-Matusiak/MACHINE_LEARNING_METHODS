# Needed libraries
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def plot_function(X,Y_train,x_label,y_label,title):
    plt.scatter(X,Y_train,label=title, s=15)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

def print_formula(model):
    print("Formula:")
    print(f"{model.intercept_[0]} + ",end="")
    for i,coef in enumerate(model.coef_[0]):
        if i > 0:
            print(f"{coef}*x^{i}",end="")
            if i < model.coef_[0].size-1:
                print(" + ",end="")
            else:
                print()

def print_results(model,model_name,X,x_poly,Y_exact,Y_pred):
    print(f"\n\n=============={model_name}==============")
    print(model.get_params())
    print("Score = ",model.score(x_poly,Y_exact))
    print("r2_score = ",r2_score(Y_exact,Y_pred))
    print("Coefficients = ",model.coef_)
    print("Intercepts = ",model.intercept_)
    print("Equation = ",end="")
    print("Mean squared error = ",mean_squared_error(Y_exact,Y_pred))


#   -------------------------------------------------------------------------------
if __name__ == "__main__":

    # plt.style.use('ggplot')
    AMOUN = 1000000
    X = 0.1 * np.linspace(-10,10,AMOUN).reshape(AMOUN,1)
    X_train, X_test = train_test_split(X,train_size=0.5)
    # Not sure if there is a need to have separated x_train and x_test
    # X_train = X_test

    Y_train = 3*X_train**3  + 0.5*X_train**2 + X_train + 2 + np.random.randn(int(AMOUN/2),1)
    Y_exact = 3*X_test**3  + 0.5*X_test**2 + X_test + 2 + np.random.randn(int(AMOUN/2),1)
    # Not sure if it should be with noise or not
    # Y_exact = 3*X_train**3  + 0.5*X_train**2 + X_train + 2



    degrees = range(1,7)
    lreg_3 = []
    y_preds = []
    MSE = []
    for deg in degrees:
        poly = PolynomialFeatures(degree=deg)
        x_poly = poly.fit_transform(X_train)
        x_poly_test = poly.fit_transform(X_test)
        lr = LinearRegression()
        lr.fit(x_poly,Y_train)
        y_lr = lr.predict(x_poly_test)
        lreg_3 = lr
        y_preds.append(y_lr)
        MSE.append(mean_squared_error(Y_exact,y_lr))
        if deg == 3:
            print_formula(lr)
            # print_results(lr,"LinearRegression",X_test,x_poly,Y,y_lr)

#   -------------------------------------------------------------------------------
    plot_function(X_train, Y_train,"X values","Y values","Exact values")
    for i,y in enumerate(y_preds):
        plot_function(X_test, y,"X values","Predicted values", f"Prediction for polynomial of {i+1} degree")
    plt.legend()
    plt.show()

#   -------------------------------------------------------------------------------
    plt.plot(degrees,MSE)    
    plt.xlabel("degrees")
    plt.ylabel("Mean squared error")
    plt.title("MSE(degree)")
    plt.grid(True)
    plt.legend()
    plt.show()

#   -------------------------------------------------------------------------------
    print(f"Mininal MSE value MSE={np.amin(MSE)} for polynomial of {np.argmin(MSE)+1} degree")