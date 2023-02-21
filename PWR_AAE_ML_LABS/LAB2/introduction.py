try:
    from cProfile import label
    import numpy as np
    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
except:
    print("Something went wrong")



def read_df_from_csv(file_name):
    path = os.getcwd()+ "\\LAB2\\" + file_name
    df = pd.read_csv(path, header=None)
    df.columns=['X','Y','label']
    return df

def plot_scatter_df(df):
    df.plot.scatter(x='X',y='Y',c='label', colormap='viridis')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Data from .csv file")
    plt.show()

if __name__ == "__main__":
    #print(os.getcwd())
    plt.style.use('ggplot')
    df = read_df_from_csv('data.csv')
    print(df)
    plot_scatter_df(df)

