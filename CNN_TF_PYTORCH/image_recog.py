# Needed libraries
try:
    from image_recog_tensorflow import *
    from image_recog_pytorch import *
except:
    print("Something went wrong")


# ------------------------------------------------------------------------------------------------
# Main block
if __name__ == "__main__":

    """
    Comparison of pretrained ResNet models for PyTorch and TensorFlow 
    """
    Pipeline_TensorF(no_epochs = 10, validation_split = 0.3, learning_rate = 0.003, batch_size = 34)
    Pipeline_PyTorch(no_epochs = 10, validation_split = 0.2, learning_rate = 0.003, batch_size = 34)

    """
    Tensorfow for various number of learning rage
    """
    Pipeline_TensorF(no_epochs = 10, validation_split = 0.3, learning_rate = 0.0001, batch_size = 34)
    Pipeline_TensorF(no_epochs = 10, validation_split = 0.3, learning_rate = 0.001, batch_size = 34)
    Pipeline_TensorF(no_epochs = 10, validation_split = 0.3, learning_rate = 0.01, batch_size = 34)
    Pipeline_TensorF(no_epochs = 10, validation_split = 0.3, learning_rate = 0.1, batch_size = 34)
    
    """
    Tensorfow for various number of learning rage
    """
    Pipeline_PyTorch(no_epochs = 10, validation_split = 0.3, learning_rate = 0.0001, batch_size = 34)
    Pipeline_PyTorch(no_epochs = 10, validation_split = 0.3, learning_rate = 0.001, batch_size = 34)
    Pipeline_PyTorch(no_epochs = 10, validation_split = 0.3, learning_rate = 0.01, batch_size = 34)
    Pipeline_PyTorch(no_epochs = 10, validation_split = 0.3, learning_rate = 0.1, batch_size = 34)
    
    """
    Another tests... 
    """
    # Pipeline_TensorF(no_epochs = 10, validation_split = 0.3, learning_rate = 0.003, batch_size = 34)
    # Pipeline_PyTorch(no_epochs = 10, validation_split = 0.2, learning_rate = 0.003, batch_size = 34)
