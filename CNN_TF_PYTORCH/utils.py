# Needed libraries
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import tensorflow as tf
    from pathlib import Path
    import seaborn as sb
    import os
except:
    print("Something went wrong")


class Dataset:
    """
    Generates dataset according to path. Contain informations like number of
    labels, training images, validation images, vector of labels etc.
    Args:
        dataset_path: String. Path to images' directory, that is the place where
            images will be fetched.
        image_df: Float. Fraction of images reserved for validation (strictly 
            between 0 and 1).
    """
    def __init__(self, dataset_path, validation_split, batch_size):
        self.dataset_path = dataset_path
        self.image_df = self.importDataset()

        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

        self.train_images = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(224, 224),
            batch_size=batch_size,
            shuffle=False,
            class_mode='categorical',
            subset='training')
        self.train_labels = self.train_images.labels

        self.validation_images = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(224, 224),
            batch_size=batch_size,
            shuffle=False,
            class_mode='categorical',
            subset='validation')    
        self.validation_labels = self.validation_images.labels

        self.number_of_labels = len(self.train_images.class_indices.keys())

    def showExampleImages(self):
        random_index = np.random.randint(0, len(self.image_df), 16)
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                                subplot_kw={'xticks': [], 'yticks': []})

        images_list = []
        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(self.image_df.Filepath[random_index[i]]))
            ax.set_title(self.image_df.Label[random_index[i]])
        plt.tight_layout()
        plt.show()

    def importDataset(self):
        image_dir = Path(self.dataset_path)
        filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + \
            list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG'))
        labels = list(map(lambda x: os.path.split(
            os.path.split(x)[0])[1], filepaths))
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')
        return pd.concat([filepaths, labels], axis=1)