# Needed libraries
try:
    from NNModel import *
except:
    print("Something went wrong")


# ------------------------------------------------------------------------------------------------
# Main block
if __name__ == "__main__":
    model_path = 'model_overfitting.h'
    dataset_path = os.getcwd() + "/MINIPROJECT/SEA_ANIMALS"
    SeaAnimalsDataset = Dataset(dataset_path=dataset_path)

    # ------------------------------------------------------------------------------------------------
    # Show example images from sea animals dataset
    if 0:
        SeaAnimalsDataset.showExampleImages()

    # ------------------------------------------------------------------------------------------------
    # Creating model
    CNN = NNModel(
        number_of_labels=SeaAnimalsDataset.number_of_labels,
        model_path = None,
        use_transfer_learning = True,
        pretrained_model = "MobileNet"
    )

    CNN.addLayer(Flatten())
    # CNN.addLayer(Dense(300, activation="relu"))
    # CNN.addLayer(Dense(250, activation="relu"))
    # CNN.addLayer(Dense(100, activation="relu"))
    CNN.addLayer(Dense(100, activation="relu"))
    CNN.addLayer(tf.keras.layers.Dropout(0.2))
    CNN.addOutputLayer()

    CNN.compileModel(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(), 
        metrics=['accuracy']
    )

    # ------------------------------------------------------------------------------------------------
    # Fit model

    if CNN.is_model_loaded == False:
        print("Number of step per epoch = ",len(SeaAnimalsDataset.train_images))
        print("Number of train images = ", len(SeaAnimalsDataset.train_labels))
        print("Number of validate images = ", len(SeaAnimalsDataset.validation_labels))
        epochs = 5

        print(CNN.model.summary())
        CNN.trainModel(SeaAnimalsDataset.train_images,
            steps_per_epoch=len(SeaAnimalsDataset.train_images),
            validation_data=SeaAnimalsDataset.validation_images,
            validation_steps=len(SeaAnimalsDataset.validation_images),
            epochs=epochs
        )
        CNN.saveModel(model_path)
        CNN.plotHistory()
    else:
        print("Model has been loaded from specific path. No need to train")

    CNN.printCrosstab(SeaAnimalsDataset.validation_images, SeaAnimalsDataset.validation_labels)
    CNN.printCrosstab(SeaAnimalsDataset.train_images, SeaAnimalsDataset.train_labels)
