try:
    from cProfile import label
    import numpy as np
    import time
    import graphviz
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    import os
except:
    print("Something went wrong")



def read_df_from_csv(file_name):
    path = os.getcwd() + "/LAB2/" + file_name
    df = pd.read_csv(path, header=None)
    df.columns=['X','Y','label']
    return df

def plot_scatter_df(df):
    df.plot.scatter(x='X',y='Y',c='label', colormap='viridis')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Data from .csv file")

def plot_decision_boundary(model, dataframe,y_train):
    x_test = np.linspace(0.0, 1.0, 1000)
    y_test = np.linspace(0.0, 1.0, 1000)
    xx, yy = np.meshgrid(x_test, y_test)
    y_mesh_predict = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.imshow(y_mesh_predict,extent=(0, 1, 0, 1),
        cmap="Wistia", origin="lower")
    plt.scatter(dataframe["X"], dataframe["Y"], 
        c=y_train, cmap="viridis")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Data plot with decision bounary")
    plt.show()

if __name__ == "__main__":
    plt.style.use('ggplot')
    df = read_df_from_csv('data.csv')
    print(df.sample(n=5))

    x_train = df.drop(columns='label')
    y_train = df['label']

    svm = SVC()
    svm.fit(x_train,y_train)

    y_pred = svm.predict(x_train)
    acc_overfitted = accuracy_score(y_train,y_pred)
    print(f"Accuracy score for defaults = {acc_overfitted}")

    plot_decision_boundary(svm, df, y_train)


    # ------------------------------------------------------------------
    # Plot accuracy vs C
    C = []
    ACC = []
    for c in np.arange(0.2,20,0.2):
        C.append(c)
        svm = SVC(C=c)
        svm.fit(x_train,y_train)

        y_pred = svm.predict(x_train)
        ACC.append(accuracy_score(y_train,y_pred))


        # ------------------------------------------------------------------
        # Plot data and its decision boundary
        if c==16:
            x_test = np.linspace(0.0, 1.0, 1000)
            y_test = np.linspace(0.0, 1.0, 1000)
            xx, yy = np.meshgrid(x_test, y_test)
            y_mesh_predict = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.imshow(y_mesh_predict,extent=(0, 1, 0, 1),
                cmap="Wistia", origin="lower")
            plt.scatter(df["X"], df["Y"], 
                c=y_train, cmap="viridis")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"SVM for C={c}")
            plt.show()

    plt.plot(C,ACC)
    plt.xlabel("C")
    plt.ylabel("accuracy")
    plt.title("accuracy vs C")
    plt.show()

    # ------------------------------------------------------------------
    # Plot accuracy vs degree
    D = []
    ACC = []
    for degree in np.arange(1,8,1):
        D.append(degree)
        svm = SVC(degree=degree, kernel = 'poly')
        svm.fit(x_train,y_train)

        y_pred = svm.predict(x_train)
        ACC.append(accuracy_score(y_train,y_pred))


        # ------------------------------------------------------------------
        # Plot data and its decision boundary
        # if c==16:
        #     x_test = np.linspace(0.0, 1.0, 1000)
        #     y_test = np.linspace(0.0, 1.0, 1000)
        #     xx, yy = np.meshgrid(x_test, y_test)
        #     y_mesh_predict = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        #     plt.imshow(y_mesh_predict,extent=(0, 1, 0, 1),
        #         cmap="Wistia", origin="lower")
        #     plt.scatter(df["X"], df["Y"], 
        #         c=y_train, cmap="viridis")
        #     plt.xlabel("X")
        #     plt.ylabel("Y")
        #     plt.title(f"SVM for C={c}")
        #     plt.show()

    plt.plot(D,ACC)
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("accuracy vs degree")
    plt.show()

    # ------------------------------------------------------------------
    # Plot accuracy vs gamma
    G = []
    ACC = []
    for index, gamma in enumerate(['scale', 'auto']):
        G.append(index)
        svm = SVC(gamma = gamma, kernel='rbf')
        svm.fit(x_train,y_train)

        y_pred = svm.predict(x_train)
        ACC.append(accuracy_score(y_train,y_pred))

        # ------------------------------------------------------------------
        # Plot data and its decision boundary
        if False:
            x_test = np.linspace(0.0, 1.0, 1000)
            y_test = np.linspace(0.0, 1.0, 1000)
            xx, yy = np.meshgrid(x_test, y_test)
            y_mesh_predict = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.imshow(y_mesh_predict,extent=(0, 1, 0, 1),
                cmap="Wistia", origin="lower")
            plt.scatter(df["X"], df["Y"], 
                c=y_train, cmap="viridis")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"SVM for C={c}")
            plt.show()

    x_gamma = ['scale', 'auto']
    plt.bar(x_gamma, ACC, width = 0.4)
    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    plt.title("accuracy vs gamma")
    plt.show()

    # ------------------------------------------------------------------
    # Plot accuracy vs kernel
    I = []
    ACC = []
    for index,kernel in enumerate(['linear', 'poly', 'rbf', 'sigmoid']):
        I.append(index)
        svm = SVC(kernel=kernel)
        svm.fit(x_train,y_train)

        y_pred = svm.predict(x_train)
        ACC.append(accuracy_score(y_train,y_pred))

        # ------------------------------------------------------------------
        # Plot data and its decision boundary
        if False:
            x_test = np.linspace(0.0, 1.0, 1000)
            y_test = np.linspace(0.0, 1.0, 1000)
            xx, yy = np.meshgrid(x_test, y_test)
            y_mesh_predict = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.imshow(y_mesh_predict,extent=(0, 1, 0, 1),
                cmap="Wistia", origin="lower")
            plt.scatter(df["X"], df["Y"], 
                c=y_train, cmap="viridis")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"SVM for C={c}")
            plt.show()

    x_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    plt.bar(x_kernels, ACC, width = 0.4)
    plt.xlabel("kernel")
    plt.ylabel("accuracy")
    plt.title("accuracy vs kernel")
    plt.show()



    # ------------------------------------------------------------------
    # Plot for GridSearchCV
    C = np.arange(0.5,20,0.5)
    kernel = ['sigmoid','poly', 'rbf','linear']
    degree = range(1,6)
    gamma = ['scale', 'auto'] 
    param_grid = dict(C=C,kernel=kernel,degree=degree,gamma=gamma)

    svm = SVC()
    grid = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=6, verbose=4)
    grid.fit(x_train,y_train)
    best = grid.best_estimator_

    pred = best.predict(x_train)
    print(f"Accuracy score for best = {accuracy_score(pred,y_pred)}")
    print(f"The best parameters are {grid.best_params_} with a score of {grid.best_score_}")

    x_test = np.linspace(0.0, 1.0, 1000)
    y_test = np.linspace(0.0, 1.0, 1000)
    xx, yy = np.meshgrid(x_test, y_test)
    y_mesh_predict = best.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.imshow(y_mesh_predict,extent=(0, 1, 0, 1),
        cmap="Wistia", origin="lower")
    plt.scatter(df["X"], df["Y"], 
        c=y_train, cmap="viridis")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"SVM for C={c}")
    plt.show()
