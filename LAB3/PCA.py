try:
    from cProfile import label
    import numpy as np
    import time
    import graphviz
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    import os
except:
    print("Something went wrong")

#----------2----------
X = np.transpose(np.array([[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
                           [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]]))

#----------5----------
mean = X.mean(axis=0)
X = X - mean

#----------4---------
plt.figure(figsize=(5,5))
plt.scatter(X[:, 0], X[:,1])
#plt.show()
plt.axvline()
plt.axhline()
#plt.show()


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
u = V[:, 1]
z = V[:, 0]
 
u = np.expand_dims(u, 1)
z = np.expand_dims(z, 1)

#----------11----------
#Y1 = np.cross(X, u)
Y = X@u@u.T


#----------12----------
meanY = np.mean(Y, axis=0)
Y = Y - meanY

plt.figure(figsize=(5,5))
plt.scatter(Y[:, 0], Y[:,1])
plt.axvline()
plt.axhline()
plt.quiver(V[0, 1], V[1, 1], color=['b'], scale=5)
plt.show()

#-----------13---------
Z = X@z@z.T
meanZ = np.mean(Z, axis=0)
Z = Z - meanZ

plt.figure(figsize=(5,5))
plt.scatter(Z[:, 0], Z[:,1])
plt.axvline()
plt.axhline()
plt.quiver(V[0, 0], V[1, 0], color=['r'], scale=5)
plt.show()