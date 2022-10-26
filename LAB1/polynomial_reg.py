# Needed libraries
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

def print_results(model,model_name,X,x_poly,Y,Y_pred):
    print(f"\n\n=============={model_name}==============")
    print(model.get_params())
    print("Score = ",model.score(x_poly,Y))
    print("r2_score = ",r2_score(Y,Y_pred))
    print("Coefficients = ",model.coef_)
    print("Intercepts = ",model.intercept_)
    print("Equation = ",end="")
    print("Mean squared error = ",mean_squared_error(Y,Y_pred))


#   -------------------------------------------------------------------------------
if __name__ == "__main__":

    # plt.style.use('ggplot')
    AMOUN = 500
    X = 0.1 * np.linspace(-10,10,AMOUN).reshape(AMOUN,1)
    Y = 3*X**3  + 0.5*X**2 + X + 2 + np.random.randn(AMOUN,1)
    Y_exact = 3*X**3  + 0.5*X**2 + X + 2

    degrees = range(1,10)
    lreg_3 = []
    y_preds = []
    MSE = []
    for deg in degrees:
        poly = PolynomialFeatures(degree=deg)
        x_poly = poly.fit_transform(X)
        lr = LinearRegression()
        lr.fit(x_poly,Y)
        y_lr = lr.predict(x_poly)
        lreg_3 = lr
        y_preds.append(y_lr)
        MSE.append(mean_squared_error(Y_exact,y_lr))
        if deg == 3:
            print_formula(lr)
            # print_results(lr,"LinearRegression",X,x_poly,Y,y_lr)

#   -------------------------------------------------------------------------------
    plot_function(X, Y,"X values","Y values","Exact values")
    for i,y in enumerate(y_preds):
        plot_function(X, y,"X values","Predicted values", f"Prediction for polynomial of {i+1} degree")
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