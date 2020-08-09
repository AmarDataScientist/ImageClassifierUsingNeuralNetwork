import argparse
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import time
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import os

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default = 'flowers/test/102/image_08004.jpg', help = 'path to image to predict')
    in_args = parser.parser_args()
    return in_args

in_args = get_input_args()
image_path = in_args.dir

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath, map_location = 'cpu'):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    #model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    
    for param in model.parameters():
        param.requires_grad = False
    return model #, checkpoint['class_to_idx']
    
model = load_checkpoint('checkpoint.pth')

#image_path = 'flowers/test/102/image_08004.jpg'
img = Image.open(image_path)
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #tensor.numpy().transpose(1,2,0)
    image = test_transforms(image)
    return image 

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
        plt.title = title
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

with Image.open(image_path) as image:
    plt.imshow(image)
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    
    #Converts 2D image to 1D vector
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    
    model.eval()
    inputs = Variable(img)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits, dim = 1)
    topk = ps.topk(topk)
    
    return (e.squeeze().tolist() for e in topk)

# TODO: Display an image along with the top 5 classes
image_path = 'flowers/test/102/image_08004.jpg'
with Image.open(image_path) as image:
    plt.imshow(image)
probs,classes = predict(image_path, model, 5)
print(probs)
print(classes)
flower_names = [cat_to_name[class_names[e]] for e in classes]
print(flower_names)

def view_classify(image_path, prob, classes, mapping):
    ''' Function for viewing an image and it's predicted classes.
    '''
    image = Image.open(image_path)
    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    flower_name = mapping[img_path.split('/')[-2]]
    ax1.set_title(flower_name)
    ax1.imshow(image)
    ax1.axis('off')
    
    y_pos = np.arange(len(prob))
    ax2.barh(y_pos, prob, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_names)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_title('Class Probability')
view_classify(image_path, probs, classes, cat_to_name)