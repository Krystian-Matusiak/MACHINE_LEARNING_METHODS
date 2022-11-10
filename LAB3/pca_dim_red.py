# Needed libraries
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_function(X, Y_train, x_label, y_label, label):
    plt.scatter(X, Y_train, label=label, s=15)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.axvline()
    plt.axhline()
    plt.legend()

def print_results(model, model_name, X, x_poly, Y_exact, Y_pred):
    print(f"\n\n=============={model_name}==============")


#   -------------------------------------------------------------------------------
if __name__ == "__main__":

    plt.style.use('ggplot')
    X = np.transpose(np.array([[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
                  [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]]))

    # Centered data
    mean = np.mean(X, axis=0)
    X = X - mean

    # Plot centered data
    plot_function(X[:,0], X[:,1],"x-axis","y-axis","Given X data")
    # plt.show()

    # Covariance matrix
    C = np.cov(X.T)
    print(C)

    # Eigenpair
    D,V = np.linalg.eig(C)
    print(f"D (eigenvalues):\n {D}")
    print(f"V (eigenvectors):\n {V}")

    # Plot data with eigenvectors
    plt.quiver(V[0,0], V[1,0], color="r", scale=5)
    plt.quiver(V[0,1], V[1,1], color="b", scale=5)
    plt.show()

    # U and Z variables
    index = D.argsort()[::-1]
    D = D[index]
    V = V[:,index]

    u = V[:,0]
    z = V[:,1]
    u = np.expand_dims(u,1)
    z = np.expand_dims(z,1)

    # print(f"u = {u}")
    # print(f"z = {z}")

    # Y calculated
    Xu = np.dot(X,u)
    Y = np.dot(Xu,u.T)
    print(f"Y = {Y}")
    plot_function(Y[:,0], Y[:,1],"x-axis","y-axis","Y projection for u")

    # Z calculated
    Xz = np.dot(X,z)
    Z = np.dot(Xz,z.T)
    print(f"Z = {Z}")
    plot_function(Z[:,0], Z[:,1],"x-axis","y-axis","Y projection for z")
    plt.show()

    # PCA from sklearn
    pca = PCA(n_components=2)
    pca.fit(X)
    train_pca = pca.transform(X)
    print(train_pca)
    plot_function(train_pca[:,0], train_pca[:,1],"x-axis","y-axis","PCA from scikit-learn")
    plt.show()
