# Needed libraries
try:
    from cProfile import label
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import tensorflow as tf
    from pathlib import Path
    import os
except:
    print("Something went wrong")


# Show example image from cifar10
def show_cifar_image():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(x_train.shape)
    print(x_train[0, 0])


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


if __name__ == "__main__":
    image_df = import_dataset()
    print(image_df)

    random_index = np.random.randint(0, len(image_df), 16)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []})

    # Show example images from sea animals dataset
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
        ax.set_title(image_df.Label[random_index[i]])
    plt.tight_layout()
    plt.show()
