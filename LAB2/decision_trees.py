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



    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(df.drop(['label'],axis="columns"),df['label'])
    tree.plot_tree(dec_tree)
    plt.show()

    y_pred = dec_tree.predict(df.drop(['label'],axis="columns"))
    acc = accuracy_score(df['label'],y_pred)
    print(f"Accuracy score = {acc}")
    
    # print(Z)
    # plt.imshow(Z, 
    #         interpolation="nearest",
    #         extent=(0, 0.3, 0, 1),
    #         cmap="Wistia", origin="lower")

    # DecisionBoundaryDisplay.from_estimator(
    #     dec_tree,
    #     df.drop(['label'],axis="columns"),
    #     cmap=plt.cm.RdYlBu,
    #     response_method="predict",
    # )
    
    labels = df.drop(['label'],axis="columns")
    print("z = " , labels)
    plt.contourf(df['X'], df['Y'], df.drop(['label'],axis="columns"), alpha=0.4)
    plt.imshow(df.to_numpy())
    plot_scatter_df(df)
    plt.show()
