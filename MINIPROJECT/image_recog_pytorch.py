import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
from torch.utils.data.sampler import SubsetRandomSampler
from enum import Enum
import seaborn as sb
import pandas as pd


def load_split_train_test(datadir, batch_size, valid_size=0.2):
    train_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                           transforms.ToTensor(),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          ])
    train_data = datasets.ImageFolder(datadir,
                                      transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                                     transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=batch_size)
    return trainloader, testloader

def predict_image(image, transforms, model, device):
    image_tensor = transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_image = image_tensor
    input_image = input_image.to(device)
    output = model(input_image)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num, data):
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data,
                                            sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = next(dataiter)
    return images, labels

def Pipeline_PyTorch(no_epochs = 10, validation_split = 0.2, learning_rate = 0.003, batch_size = 34):
    # region ---------------------------------------------- Load data
    is_model_loaded = False
    data_dir =[]
    model_path = 'ModelPytorch.pth'

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

    trainloader, testloader = load_split_train_test(data_dir, batch_size=batch_size, valid_size=validation_split)
    print(f"All classes: {trainloader.dataset.classes}")
    print(f"Number of batches for train data: {len(trainloader)}")
    print(f"Number of batches for test data: {len(testloader)}")
    # endregion

    # region ---------------------------------------------- Creating model
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(device)
    model = models.resnet50(pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 100),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(100, 7),
                             nn.LogSoftmax(dim=1))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    model.to(device)
    # endregion

    #region ---------------------------------------------- Training
    if is_model_loaded:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path)
    else:
        epochs = no_epochs
        train_losses, test_losses = [], []
        train_acc, test_acc = [], []        

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}.. ")
            iteration = 0
            
            train_accuracy = 0
            test_accuracy = 0

            train_loss = 0
            test_loss = 0

            for inputs, labels in trainloader:
                iteration += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                predicts = model(inputs)
                loss = criterion(predicts, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                ps = torch.exp(predicts)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                train_accuracy += torch.mean(
                    equals.type(torch.FloatTensor)).item()

            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(
                        device), labels.to(device)

                    predicts = model(inputs)
                    batch_loss = criterion(predicts, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(predicts)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_accuracy += torch.mean(
                        equals.type(torch.FloatTensor)).item()
            model.train()

            train_losses.append(train_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))

            train_acc.append(train_accuracy/len(trainloader))
            test_acc.append(test_accuracy/len(testloader))
            print(f"Train loss: {train_loss/len(trainloader):.4f}.. "
                  f"Test loss: {test_loss/len(testloader):.4f}.. "
                  f"Train accuracy: {train_accuracy/len(trainloader):.4f}.. "
                  f"Test accuracy: {test_accuracy/len(testloader):.4f}")

        torch.save(model, model_path)

        plt.figure()
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(train_acc, label='Training accuracy')
        plt.plot(test_acc, label='Validation accuracy')
        plt.legend(frameon=False)
        plt.grid()
        plt.show()
    # endregion

    #region ---------------------------------------------- Testing model
    model.eval()
    test_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          ])
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes

    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(batch_size, data)
    fig = plt.figure(figsize=(10, 10))
    for ii in range(9):
        image = to_pil(images[ii])
        index = predict_image(image, test_transforms, model, device)
        sub = fig.add_subplot(4, 3, ii+1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]))
        plt.axis('off')
        plt.imshow(image, )
    plt.show()
    # endregion

    #region ---------------------------------------------- Crosstab
    predicts_vec = np.array([])
    labels_vec = np.array([])
    for inputs, labels in testloader:
        inputs, labels = inputs.to(
            device), labels.to(device)

        predicts = model(inputs)
        predicts = predicts.max(1).indices

        labels = labels.data.cpu().numpy()
        predicts = predicts.data.cpu().numpy()
        predicts_vec = np.append(predicts_vec,predicts)
        labels_vec = np.append(labels_vec,labels)

    data = {'Exact_values': labels_vec, "Predictions": predicts_vec}
    df = pd.DataFrame(data=data)
    print(df)

    results = pd.crosstab(df['Exact_values'],df['Predictions'])
    plt.figure(figsize=(10,7))
    sb.heatmap(results, annot=True, cmap="OrRd", fmt=".0f")
    # endregion 


if __name__ == "__main__":
    Pipeline_PyTorch(no_epochs = 10, validation_split = 0.2, learning_rate = 0.003, batch_size = 34)
