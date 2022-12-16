# Needed libraries
try:
    from NNModel import *
except:
    print("Something went wrong")


def Pipeline(TransferLearningNetworkName, no_epochs, validation_split):
    model_path = 'model.h'
    dataset_path = os.getcwd() + "/MINIPROJECT/SEA_ANIMALS"
    SeaAnimalsDataset = Dataset(dataset_path=dataset_path, validation_split=validation_split)

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
        pretrained_model = TransferLearningNetworkName
    )
    CNN.addLayer(Flatten())
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
        epochs = no_epochs

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



# ------------------------------------------------------------------------------------------------
# Main block
if __name__ == "__main__":

    """
    Test pretrained models for high number of epocks to see how the model behaves and
    when they start to be overfitted.    
    """
    Pipeline(TransferLearningNetworkName="MobileNet", no_epochs=30, validation_split=0.3)
    Pipeline(TransferLearningNetworkName="ResNet", no_epochs=30, validation_split=0.3)


    """
    Test better pretrained model for various value of validation split (validation images
    to all images ratio) using constant (optimal) number of epochs.
    """
    Pipeline(TransferLearningNetworkName="MobileNet", no_epochs=10, validation_split=0.1)
    Pipeline(TransferLearningNetworkName="MobileNet", no_epochs=10, validation_split=0.2)
    Pipeline(TransferLearningNetworkName="MobileNet", no_epochs=10, validation_split=0.3)
    Pipeline(TransferLearningNetworkName="MobileNet", no_epochs=10, validation_split=0.4)
    Pipeline(TransferLearningNetworkName="MobileNet", no_epochs=10, validation_split=0.7)
   

    """
    Another test... 
    """
    # Pipeline(TransferLearningNetworkName="", no_epochs=, validation_split=)