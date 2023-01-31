# Needed libraries
try:
    from utils import *
    from tensorflow.keras.layers import Dense, Flatten

except:
    print("Something went wrong")





class NNModel:
    """
    Creates Tensorflow neural network model. Contains significant functions that provides
    various features like plot accuracy, print crosstab etc.
    Args:
        model_path: String. Path to prepared model. If None creates another model. 
        number_of_labels: Integer. Contains number of labels, that is number of output neurons.
        use_transfer_learning: Boolean. Flag which determines if input layer will be pretrained
            model or not.
        pretrained_model: String. Deterimes which pretrained model will be used.
    """
    def __init__(self, number_of_labels, model_path = None, use_transfer_learning = True, pretrained_model = "MobileNet"):
        self.is_model_loaded = False
        if model_path == None:
            self.input_layer = None

            if use_transfer_learning:
                if pretrained_model == "MobileNet":
                    self.input_layer = tf.keras.applications.MobileNetV2(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights='imagenet',
                        pooling='avg'
                    )
                    self.input_layer.trainable = False
                elif pretrained_model == "ResNet":
                    self.input_layer = tf.keras.applications.ResNet50(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights='imagenet',
                        pooling='avg'
                    )
                    self.input_layer.trainable = False
                else:
                    pass

            if self.input_layer == None:
                self.input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
            self.model = tf.keras.Sequential([
                self.input_layer
            ])
            self.number_of_labels = number_of_labels
        else:
            self.model = self.loadModel(model_path)
            self.is_model_loaded = True

        self.history = None

    def addLayer(self, layer):
        if self.is_model_loaded:
            print("Cannot add layer - model has been loaded from specific path.")
        else:
            self.model.add(layer)

    def addOutputLayer(self):
        if self.is_model_loaded:
            print("Cannot add layer - model has been loaded from specific path.")
        else:
            self.addLayer(Dense(self.number_of_labels, activation="softmax"))

    def compileModel(self, loss, optimizer, metrics):
        if self.is_model_loaded:
            print("Cannot compile - model has been loaded from particular path.")
        else:
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def loadModel(self, model_path):
        return tf.keras.models.load_model(model_path)

    def saveModel(self, model_path):
        self.model.save(model_path)

    def trainModel(self, train_data, steps_per_epoch, validation_data, validation_steps, epochs):
        if self.is_model_loaded:
            print("Cannot train - model has been loaded from specific path.")
        else:
            self.history = self.model.fit(
                train_data, 
                steps_per_epoch=steps_per_epoch, 
                validation_data=validation_data, 
                validation_steps=validation_steps, 
                epochs=epochs
            )

    def plotHistory(self):
        plt.figure()
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','validate'], loc='upper left')
        # plt.show()

        plt.figure()
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validate'], loc='upper left')
        # plt.show()

    def printCrosstab(self, X_test, Y_test):
        labels = list(X_test.class_indices.keys())
        preds = self.model.predict(X_test)
        y_preds = tf.argmax(preds, 1).numpy()

        exact_vec = []
        for y in Y_test:
            exact_vec.append(labels[y])
        predict_vec = []
        for y in y_preds:
            predict_vec.append(labels[y])

        data = {'Exact_values': exact_vec, "Predictions": predict_vec}
        df = pd.DataFrame(data=data)
        print(df)

        results = pd.crosstab(df['Exact_values'],df['Predictions'])
        print(results)
        plt.figure(figsize=(10,7))
        sb.heatmap(results, annot=True, cmap="OrRd", fmt=".0f")
        plt.title("Crosstab for tensorflow")
        # plt.show()