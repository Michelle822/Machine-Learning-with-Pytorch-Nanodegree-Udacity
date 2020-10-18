# Imports here
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
parsers = argparse.ArgumentParser()
parsers.add_argument('data_dir', action="store")
parsers.add_argument('save_dir', action="store")
parsers.add_argument('--top_k', dest="top_k", default=5)
parsers.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')

parsers.add_argument('--gpu', action="store_const", dest="device", const="gpu", default='cpu')

options = parsers.parse_args()
data_dir = options.data_dir
save_dir = options.save_dir
top_k = options.top_k
category_names = options.category_names
device = options.device

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint (filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False

    model.class_to_idx = checkpoint['class_to_idx']

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
    model.load_state_dict(checkpoint['state_dict'])

    return model

model = load_checkpoint('checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
        ])

    image = img_transforms(Image.open(image))

    return image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''



    model.to("cpu")


    model.eval();

    # Converion of image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Probabilities  by passing through the function
    log_ps = model.forward(torch_image)

    # Conversion
    linear_probs = torch.exp(log_ps)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)

    # Detach all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    # Convert to classes
    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers
    # TODO: Implement the code to predict the class from an image file

image_path = "flowers/test/10/image_07090.jpg"

# Set up plot
plt.figure(figsize = (15,6))
ax = plt.subplot(1,2,1)

# Set up title
flower_num = image_path.split('/')[2]
title_ = cat_to_name[flower_num]

# Plot flower
img = process_image(image_path)
imshow(img, ax, title = title_);

# Make prediction
probs, labs, flowers = predict(image_path, model)

# Plot bar chart
plt.subplot(1,2,2)
sns.barplot(x=probs, y=flowers, color=sns.color_palette()[1]);
plt.show()
