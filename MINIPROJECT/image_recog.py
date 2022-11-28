# Needed libraries
try:
    from cProfile import label
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras import optimizers, losses
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
    data = os.getcwd() + "/MINIPROJECT/SEA_ANIMALS"

    random_index = np.random.randint(0, len(image_df), 16)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []})

    # Show example images from sea animals dataset
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
        ax.set_title(image_df.Label[random_index[i]])
    plt.tight_layout()
    plt.show()

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    train_images = train_datagen.flow_from_directory(
        data,
        target_size=(224, 224),
        batch_size=34,
        class_mode='categorical',
        subset='training')  # set as training data

    mobile_model = Sequential()

    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False

    mobile_model.add(pretrained_model)
    mobile_model.add(Flatten())
    mobile_model.add(Dense(212, activation='relu'))
    mobile_model.add(Dropout(0.2))
    mobile_model.add(Dense(7, activation='softmax'))

    mobile_model.compile(loss='categorical_crossentropy',
                         optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    history = mobile_model.fit(train_images,
                               steps_per_epoch=len(train_images),
                               epochs=1)

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    print(2)
    print(2)