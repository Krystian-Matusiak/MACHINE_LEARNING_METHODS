# Needed libraries
try:
    from NNModel import *
    from enum import Enum
except:
    print("Something went wrong")


def Pipeline_TensorF(no_epochs = 10, validation_split = 0.2, learning_rate = 0.003, batch_size = 34, TransferLearningNetworkName="ResNet"):
    class Env(Enum):
        LINUX = 1
        WINDOWS = 2
        GOOGLE_COLAB = 3

    env = Env.LINUX

    if env == Env.LINUX:
        data_dir = os.getcwd() + "/MINIPROJECT/SEA_ANIMALS"
    elif env == Env.WINDOWS:
        data_dir = 'D:\\studia\\II_stopien\\2sem\\ML_L\\repo\\MACHINE_LEARNING_METHODS\\MINIPROJECT\\SEA_ANIMALS'
    elif env == Env.GOOGLE_COLAB:
        data_dir = "./SEA_ANIMALS/SEA_ANIMALS"
    
    dataset_path = data_dir
    SeaAnimalsDataset = Dataset(dataset_path=dataset_path, validation_split=validation_split, batch_size=batch_size)
    model_path = 'ModelTensorflow.pth'

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
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
    plt.show()


# ------------------------------------------------------------------------------------------------
# Main block
if __name__ == "__main__":

    """
    Test pretrained models for high number of epocks to see how the model behaves and
    when they start to be overfitted.    
    """
    Pipeline_TensorF(no_epochs=7, validation_split=0.3, learning_rate=0.003)
    Pipeline_TensorF(no_epochs=30, validation_split=0.3, learning_rate=0.003)

    """
    Test better pretrained model for various value of validation split (validation images
    to all images ratio) using constant (optimal) number of epochs.
    """
    Pipeline_TensorF(no_epochs=10, validation_split=0.1, learning_rate=0.003)
    Pipeline_TensorF(no_epochs=10, validation_split=0.2, learning_rate=0.003)
    Pipeline_TensorF(no_epochs=10, validation_split=0.3, learning_rate=0.003)
    Pipeline_TensorF(no_epochs=10, validation_split=0.4, learning_rate=0.003)
    Pipeline_TensorF(no_epochs=10, validation_split=0.7, learning_rate=0.003)
   

    """
    Another test... 
    """
    # Pipeline_TensorF(TransferLearningNetworkName="", no_epochs=, validation_split=)