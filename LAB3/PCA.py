try:
    from cProfile import label
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import os
except:
    print("Something went wrong")

def plot_function(X, Y_train, x_label, y_label, label,alpha=1.0):
    plt.scatter(X, Y_train, label=label, s=15, alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.axvline()
    plt.axhline()
    plt.legend()

if __name__ == "__main__":
    #----------2----------
    X = np.transpose(np.array([[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
                            [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]]))

    #----------5----------
    mean = X.mean(axis=0)
    X = X - mean

    #----------4---------
    plt.figure(figsize=(5,5))
    plot_function(X[:, 0], X[:,1],"x-axis","y-axis","Given X data", 0.2)

    # plt.show()

    #----------6----------
    cov = np.cov(X.T)
    print("Covariance matrix:", cov)

    #----------7-----------
    [W, V] = np.linalg.eig(cov)
    print("Eigenvalues:", W)
    print("Eigenvectors:", V)

    #----------9-----------
    plt.quiver(V[0, 0], V[1, 0], color=['r'], scale=5)
    plt.quiver(V[0, 1], V[1, 1], color=['b'], scale=5)
    plt.show()

    #----------10-----------
    index = W.argsort()[::-1]
    W = W[index]
    V = V[:,index]

    u = V[:, 0]
    z = V[:, 1]
    
    u = np.expand_dims(u, 1)
    z = np.expand_dims(z, 1)

    #----------11----------
    #Y1 = np.cross(X, u)
    Y = X@u@u.T


    #----------12----------
    plt.figure(figsize=(5,5))
    plot_function(Y[:,0], Y[:,1],"x-axis","y-axis","Y projection for u",0.95)
    plt.quiver(V[0, 0], V[1, 0], color=['b'], scale=5)
    plt.show()

    #-----------13---------
    Z = X@z@z.T

    plt.figure(figsize=(5,5))
    plot_function(Z[:, 0], Z[:,1],"x-axis","y-axis","Y projection for z",0.95)
    plt.quiver(V[0,1], V[1,1], color=['r'], scale=5)
    plt.show()

    # -----------------------------------------------------------
    # PCA plot for both vectors
    plt.figure(figsize=(5,5))
    plot_function(Y[:,0], Y[:,1],"x-axis","y-axis","Y projection for u",0.95)
    plot_function(Z[:, 0], Z[:,1],"x-axis","y-axis","Y projection for z",0.95)
    plt.quiver(V[0, 0], V[1, 0], color=['b'], scale=5)
    plt.quiver(V[0,1], V[1,1], color=['r'], scale=5)    
    plt.title("Plot for both eigenvectors")
    # plt.show()

    # -----------------------------------------------------------
    # PCA from sklearn
    plt.figure(figsize=(5,5))
    pca = PCA(n_components=2)
    pca.fit(X)
    Vectors = pca.components_
    print(f"Vectors = {Vectors}")

    u = Vectors[:,0]
    z = Vectors[:,1]
    u = np.expand_dims(u,1)
    z = np.expand_dims(z,1)

    # Y calculated
    Y = X@u@u.T
    Z = X@z@z.T
    plot_function(Y[:,0], Y[:,1],"x-axis","y-axis","Y projection for u",0.95)
    plot_function(Z[:,0], Z[:,1],"x-axis","y-axis","Y projection for z", 0.95)
    plt.title("PCA from sci-kit learn python library")
    plt.quiver(Vectors[0,0], Vectors[1,0], color="r", scale=5)
    plt.quiver(Vectors[0,1], Vectors[1,1], color="b", scale=5)
    plt.show()
