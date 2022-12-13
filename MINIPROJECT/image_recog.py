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
    from utils import *
    import os
except:
    print("Something went wrong")


# ------------------------------------------------------------------------------------------------
# Main block
if __name__ == "__main__":
    image_df = import_dataset()
    print(image_df)
    data = os.getcwd() + "/MINIPROJECT/SEA_ANIMALS"
    number_of_labels = 5

    # ------------------------------------------------------------------------------------------------
    # Show example images from sea animals dataset
    if 0:
        random_index = np.random.randint(0, len(image_df), 16)
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                                subplot_kw={'xticks': [], 'yticks': []})

        images_list = []
        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
            ax.set_title(image_df.Label[random_index[i]])
        plt.tight_layout()
        plt.show()

    
    # ------------------------------------------------------------------------------------------------
    # Train and validate dataset from ImageDataGenerator()
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

    train_images = train_datagen.flow_from_directory(
        data,
        target_size=(224, 224),
        batch_size=34,
        shuffle=False,
        class_mode='categorical',
        subset='training')
    train_labels = train_images.labels

    validation_images = train_datagen.flow_from_directory(
        data,
        target_size=(224, 224),
        batch_size=34,
        shuffle=False,
        class_mode='categorical',
        subset='validation')    
    validation_labels = validation_images.labels


    # ------------------------------------------------------------------------------------------------
    # Creating model
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False

    mobile_model = tf.keras.Sequential([
        # tf.keras.layers.Input(shape=(224, 224, 3)),
        pretrained_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(number_of_labels, activation="softmax")
    ])


    mobile_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                         optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    # ------------------------------------------------------------------------------------------------
    # Fit model
    print("Number of step per epoch = ",len(train_images))
    print("Number of train images = ", len(train_labels))
    print("Number of validate images = ", len(validation_labels))
    epochs = 10

    save_or_load = 2
    if save_or_load == 1:
        history = mobile_model.fit(train_images,
                                steps_per_epoch=len(train_images),
                                validation_data=validation_images,
                                validation_steps=len(validation_images),
                                epochs=epochs)
        print(history.history.keys())
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','validate'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validate'], loc='upper left')
        plt.show()
        mobile_model.save("model.h")
    if save_or_load == 2:
        mobile_model = tf.keras.models.load_model('model.h')

    print_crosstab(validation_images, validation_labels, mobile_model)
    print_crosstab(train_images, train_labels, mobile_model)
