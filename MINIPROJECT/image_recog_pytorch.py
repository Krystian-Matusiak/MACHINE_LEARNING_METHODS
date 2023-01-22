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

def predict_image(image, transforms, model):
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

if __name__ == "__main__":
    # region ---------------------------------------------- Load data
    is_linux = True
    is_model_loaded = False
    data_dir = []
    model_name = 'PytorchModel.pth'

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

    BATCH_SIZE = 34
    trainloader, testloader = load_split_train_test(data_dir, batch_size=BATCH_SIZE, valid_size=0.2)
    print(f"All classes: {trainloader.dataset.classes}")
    print(f"Number of batches for train data: {len(trainloader)}")
    print(f"Number of batches for test data: {len(testloader)}")
    # endregion

    # region ---------------------------------------------- Creating model
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(device)
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 100),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(100, 7),
                             nn.LogSoftmax(dim=1))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    # endregion

    #region ---------------------------------------------- Training
    if is_model_loaded:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_name)
    else:
        epochs = 10
        train_losses, test_losses = [], []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}.. ")
            iteration = 0
            accuracy = 0
            running_loss = 0
            test_loss = 0

            for inputs, labels in trainloader:
                iteration += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                pregicts = model(inputs)
                loss = criterion(pregicts, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(
                        device), labels.to(device)

                    pregicts = model(inputs)
                    batch_loss = criterion(pregicts, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(pregicts)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(
                        equals.type(torch.FloatTensor)).item()
            model.train()

            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            print(f"Train loss: {running_loss/len(trainloader):.4f}.. "
                  f"Test loss: {test_loss/len(testloader):.4f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.4f}")

        torch.save(model, model_name)
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()
    # endregion

    #region ---------------------------------------------- Evaluation
    model.eval()
    # endregion

    #region ----------------------------------------------prediction
    test_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          ])
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    # endregion

    #region ---------------------------------------------- Test
    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(BATCH_SIZE, data)
    fig = plt.figure(figsize=(10, 10))
    for ii in range(9):
        image = to_pil(images[ii])
        index = predict_image(image, test_transforms, model)
        sub = fig.add_subplot(4, 3, ii+1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]))
        plt.axis('off')
        plt.imshow(image, )
    plt.show()
    # endregion
