try:
    from cProfile import label
    import numpy as np
    import time
    import graphviz
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
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

    dec_tree = DecisionTreeClassifier()
    # dec_tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3)
    dec_tree.fit(x_train,y_train)

    y_pred = dec_tree.predict(x_train)
    acc_overfitted = accuracy_score(y_train,y_pred)
    print(f"Accuracy score for defaults = {acc_overfitted}")
    print(f"Depth:{dec_tree.get_depth()} and leaves = {dec_tree.get_n_leaves()}")

    # ------------------------------------------------------------------
    # To plot data and its decision boundary
    plot_decision_boundary(dec_tree,df,y_train)

    # ------------------------------------------------------------------
    # To plot the whole tree
    tree.plot_tree(dec_tree)
    plt.show()

    # ------------------------------------------------------------------
    # Plot Accuracy vs depth
    D = range(1,12)
    ACC = []
    for depth in range(1,12):
        dtree = DecisionTreeClassifier(max_depth=depth)
        dtree.fit(x_train,y_train)
        y_pred = dtree.predict(x_train)
        ACC.append(accuracy_score(y_train,y_pred))

    plt.plot(D,ACC)
    plt.xlabel("depth")
    plt.ylabel("Accuracy")
    plt.title("Acc(depth)")
    plt.show()

    # ------------------------------------------------------------------
    # Plot Accuracy vs leaves samples
    LS = range(1,12)
    ACC = []
    for leaf_samples in range(1,12):
        dtree = DecisionTreeClassifier(min_samples_leaf=leaf_samples)
        dtree.fit(x_train,y_train)
        y_pred = dtree.predict(x_train)
        ACC.append(accuracy_score(y_train,y_pred))

    plt.plot(LS,ACC)
    plt.xlabel("Minimum samples leaf")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs minimum samples leaf")
    plt.show()

    # ------------------------------------------------------------------
    # Plot for depth and min samples leaf

    depth = np.arange(1,15)
    leaf_samples = np.arange(1,15)
    param_grid = dict(max_depth=depth, min_samples_leaf=leaf_samples)

    dtree = DecisionTreeClassifier()
    grid = GridSearchCV(estimator=dtree, param_grid=param_grid, scoring='accuracy',verbose=2)
    grid.fit(x_train,y_train)
    best = grid.best_estimator_

    pred = best.predict(x_train)
    print(f"The best parameters are {grid.best_params_} with a score of {accuracy_score(y_train,pred)}")
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),
                            pd.DataFrame(grid.cv_results_["mean_test_score"], 
                            columns=["ACC"])],
                            axis=1)
    grid_results.plot.scatter(x='max_depth',y='min_samples_leaf',c='ACC', colormap='viridis')
    plt.show()

    # ------------------------------------------------------------------
    # To plot data and its decision boundary for best
    plot_decision_boundary(best,df,y_train)


    # ------------------------------------------------------------------
    # Random forest case

    param_grid = {
        "n_estimators": np.arange(5,70,5)
    }

    random_forest = RandomForestClassifier()
    grid = GridSearchCV(estimator=random_forest, param_grid=param_grid, scoring='accuracy',verbose=2, cv=5)
    grid.fit(x_train,y_train)
    best = grid.best_estimator_

    pred = best.predict(x_train)
    print(f"The best parameters are {grid.best_params_} with a score of {accuracy_score(y_train,pred)}")

    plot_decision_boundary(best,df,y_train)


