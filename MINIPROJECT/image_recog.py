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
    from sklearn.model_selection import train_test_split
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

# Print Cross table
def print_crosstab(X_test, Y_test, model):
    y_true = tf.argmax(Y_test, 1)
    y_pred = tf.argmax(model.predict(X_test), 0)
    results = pd.crosstab(index=y_true, columns=y_pred)
    print(results)

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

# ------------------------------------------------------------------------------------------------
# Main block
if __name__ == "__main__":
    image_df = import_dataset()
    print(image_df)
    data = os.getcwd() + "/MINIPROJECT/SEA_ANIMALS"


    # ------------------------------------------------------------------------------------------------
    # Show example images from sea animals dataset
    if 1:
        random_index = np.random.randint(0, len(image_df), 16)
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                                subplot_kw={'xticks': [], 'yticks': []})

        images_list = []
        for i in range(len(image_df.Filepath)):
            images_list.append(plt.imread(image_df.Filepath[i]))
        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
            ax.set_title(image_df.Label[random_index[i]])
        plt.tight_layout()
        plt.show()
        image_df.insert(2, "Image", images_list, True)

    
    # ------------------------------------------------------------------------------------------------
    # Train and validate dataset from ImageDataGenerator()
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',validation_split=0.2)

    train_images = train_datagen.flow_from_directory(
        data,
        target_size=(224, 224),
        batch_size=34,
        class_mode='categorical',
        subset='training')  # set as training data

    train_labels = train_images.labels
    print(f"Keys: {train_labels}")

    validation_images = train_datagen.flow_from_directory(
        data,
        target_size=(224, 224),
        batch_size=34,
        class_mode='categorical',
        subset='validation')  # set as training data
    
    validation_labels = validation_images.labels
    print(f"Keys: {validation_labels}")

    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False

    # mobile_model = Sequential()
    # # mobile_model.add(pretrained_model)
    # mobile_model.add(Input(shape=(224, 224, 3)))
    # mobile_model.add(Flatten())
    # mobile_model.add(Dense(200, activation='relu'))
    # mobile_model.add(Dense(7, activation='softmax'))


    # ------------------------------------------------------------------------------------------------
    # Own approach
    df_train, df_test = train_test_split(image_df, train_size=0.75)

    X = image_df["Image"]
    Y = image_df["Label"].tolist()

    X_test = image_df["Image"]
    Y_test = image_df["Label"].tolist()

    mobile_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dense(7, activation="softmax")
    ])

    mobile_model.compile(loss='categorical_crossentropy',
                         optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    save_or_load = 1
    if save_or_load == 1:
        history = mobile_model.fit(x=X, y=Y,
                                steps_per_epoch=10,
                                epochs=16)
        print(history.history.keys())
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        mobile_model.save("model2.h")
    if save_or_load == 2:
        mobile_model = tf.keras.models.load_model('model2.h')

    print(2)
    print_crosstab(X_test, Y_test, mobile_model)
    print(2)
