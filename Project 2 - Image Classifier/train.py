
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import argparse

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)




parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action="store")
parser.add_argument('--save_dir', action="store", dest="save_dir", default='checkpoint')
parser.add_argument('--arch', action="store", dest="arch", default='vgg16')
parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001)
parser.add_argument('--epochs', action="store", dest="epochs", default=3)
parser.add_argument('--gpu', action="store_const", dest="device", const="gpu", default='cpu')

results = parser.parse_args()
data_dir = results.data_dir
save_dir = results.save_dir
arch = results.arch
learning_rate = results.learning_rate
epochs = results.epochs
device = results.device


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
# Train
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
# Test
test_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(244),
                                    transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
# Validate
valid_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(244),
                                    transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = False)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = False)

if arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(25088,4000)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.2)),
                           ('fc2',nn.Linear(4000,2000)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.2)),
                           ('fc3',nn.Linear(2000,102)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr = 0.001)

def validation(device, model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    count = 0
    model.to(device)
    for images, labels in testloader:

        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

        count += 1

    return test_loss/count, accuracy/count

def train(device, model, epochs, print_every):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            #print(inputs)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(device, model, validloader, criterion)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss),
                      "Validation Accuracy: {:.3f}".format(accuracy),
                      "Steps: ", steps)
                running_loss = 0

def check_accuarcy_on_test(testloader, device):
    model.eval()
    with torch.no_grad():
    test_loss, accuracy = validation(device, model, testloader, criterion)
    print("Test loss = ", test_loss)
    print("Accuracy = ", accuracy)


checkpoint = {
    'state_dict': model.state_dict(),
    'class_to_idx': train_data.class_to_idx
}

torch.save(checkpoint,'checkpoint.pth')
