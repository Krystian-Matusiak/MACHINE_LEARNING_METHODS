# Needed libraries
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


def plot_function(X, Y_train, x_label, y_label, title):
    plt.scatter(X, Y_train, label=title, s=15)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)


def print_results(model, model_name, X, x_poly, Y_exact, Y_pred):
    print(f"\n\n=============={model_name}==============")


#   -------------------------------------------------------------------------------
if __name__ == "__main__":

    plt.style.use('ggplot')
    X = np.transpose(np.array([[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
                  [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]]))
    plot_function(X[:,0], X[:,1],"x-axis","y-axis","Plot of given data")
    plt.show()