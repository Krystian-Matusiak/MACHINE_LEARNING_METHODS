# Needed libraries
try:
    from cProfile import label
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import optimizers, losses
    import tensorflow as tf
    from pathlib import Path
    import seaborn as sb
    import os
except:
    print("Something went wrong")


# Show example image from cifar10
def show_cifar_image():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(x_train.shape)
    print(x_train[0, 0])

# Print Cross table
def print_crosstab(X_test, Y_test, model):
    labels = list(X_test.class_indices.keys())
    preds = model.predict(X_test)
    y_preds = tf.argmax(preds, 1).numpy()

    exact_vec = []
    for y in Y_test:
        exact_vec.append(labels[y])
    predict_vec = []
    for y in y_preds:
        predict_vec.append(labels[y])

    data = {'Exact_values': exact_vec, "Predictions": predict_vec}
    df = pd.DataFrame(data=data)
    # print(df)

    results = pd.crosstab(df['Exact_values'],df['Predictions'])
    print(results)
    plt.figure(figsize=(10,7))
    sb.heatmap(results, annot=True, cmap="OrRd", fmt=".0f")
    plt.show()

# Import data from SEA_ANIMALS directory
def import_dataset():
    path = os.getcwd() + "/MINIPROJECT/SEA_ANIMALS"
    image_dir = Path(path)
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + \
        list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG'))
    labels = list(map(lambda x: os.path.split(
        os.path.split(x)[0])[1], filepaths))
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    return pd.concat([filepaths, labels], axis=1)
