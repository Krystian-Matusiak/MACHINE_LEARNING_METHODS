try:
    from cProfile import label
    import numpy as np
    import time
    import graphviz
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.metrics import accuracy_score
except:
    print("Something went wrong")



def read_df_from_csv(file_name):
    df = pd.read_csv('data.csv',header=None)
    df.columns=['X','Y','label']
    return df

def plot_scatter_df(df):
    df.plot.scatter(x='X',y='Y',c='label', colormap='viridis')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Data from .csv file")

if __name__ == "__main__":
    plt.style.use('ggplot')
    df = read_df_from_csv('data.csv')
    print(df)

    x_train = df.drop(['label'],axis="columns")
    y_train = df['label']

    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(df.drop(columns='label'),y_train)

    y_pred = dec_tree.predict(x_train)
    acc = accuracy_score(y_train,y_pred)
    print(f"Accuracy score = {acc}")



    # ------------------------------------------------------------------
    # To plot data and its decision boundary
    x_test = np.linspace(0.0, 1.0, 1000)
    y_test = np.linspace(0.0, 1.0, 1000)
    xx, yy = np.meshgrid(x_test, y_test)
    y_mesh_predict = dec_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.imshow(y_mesh_predict, 
        interpolation="nearest",
           extent=(0, 1, 0, 1),
        cmap="Wistia", origin="lower")
    plt.scatter(df["X"], df["Y"], 
        c=df["label"], cmap="viridis")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Data plot with decision bounary")
    plt.show()

    # ------------------------------------------------------------------
    # To plot the whole tree
    tree.plot_tree(dec_tree)
    plt.show()
