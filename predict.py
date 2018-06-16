#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Chung Leuk Lee.
# DATE CREATED: 06/09/2018                                  
# REVISED DATE: 06/16/2018  #modiy and test/fail
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python predict.py --pth vgg.pth --dir flowers/valid/1/image_06739.jpg 
##
#Imports python modules
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import argparse
from time import time, sleep

def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args() 
    
    model, optimizer, loss, class_to_idx = load_checkpoint(in_arg.pth)
    
    image = process_image(in_arg.dir)
    
    predict(image, model, topk=5)
        
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

# Functions defined below
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Create command line arguments args.dir 
    parser.add_argument('--pth', type=str, default='vgg.pth', help='file path')
    parser.add_argument('--dir', type=str, default='flowers/valid/1/image_06739.jpg', help='select image dir')

    
    # returns parsed argument collection
    return parser.parse_args()

#Label
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Building and training the classifier
## We check whether GPU is available on PC or not otherwise we use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

vgg16 = models.vgg16(pretrained=True).to(device)
densenet121 = models.densenet121(pretrained=True).to(device)


# Loading the checkpoint
def load_checkpoint(filepath_pth):
    if filepath_pth == 'vgg.pth':
        checkpoint = torch.load('vgg.pth')
        reloaded_model = models.vgg16(pretrained=False).to(device)
        classifier = nn.Sequential(OrderedDict(
                          [('fc1', nn.Linear(in_features=25088,out_features=4096)), 
                           ('dropout1', nn.Dropout(0.15)),
                           ('relu1', nn.ReLU()), 
                           ('fc2',nn.Linear(in_features=4096, out_features=512)),
                           ('dropout2', nn.Dropout(0.15)), 
                           ('relu2',nn.ReLU()), 
                           ('fc3',nn.Linear(in_features=512,out_features=102)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))
        reloaded_model.classifier = classifier
        reloaded_model.load_state_dict(checkpoint['model_dict'])
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(reloaded_model.classifier.parameters(), lr=0.01, momentum=0.9)
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion.load_state_dict(checkpoint['loss'])
        reloaded_model = reloaded_model.to(device)
        return reloaded_model, optimizer, criterion, checkpoint['class_to_idx']

    else:
        checkpoint = torch.load('densenet.pth')
        reloaded_model = models.densenet121(pretrained=False).to(device)
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features=1024, out_features=512)),
                          ('dropout1',nn.Dropout(0.5)),
                          ('relu1', nn.ReLU()), 
                          ('fc2', nn.Linear(in_features=512, out_features=102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        reloaded_model.classifier = classifier
        reloaded_model.load_state_dict(checkpoint['model_dict'])
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(
        reloaded_model.classifier.parameters(), lr=0.01, momentum=0.9)
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion.load_state_dict(checkpoint['loss'])
        reloaded_model = reloaded_model.to(device)
        return reloaded_model, optimizer, criterion, checkpoint['class_to_idx']
                

def process_image(image_dir,normalize=True):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_dir)
    img = img.convert('RGB')
    img = np.array(img.resize((256,256)).crop((16,16,240,240))) ##Cropping image from center to get (224,224)
    to_tensor = transforms.ToTensor() ## Transoforming image to tensor so that image with pixel values 
    img = to_tensor(img)              ## between 0-255 gets transformed to 0-1 floats which our model expects.
    img = img.numpy()       ## Converting to numpy array fromm pytorch tensor for normalizatioin operation below.
    #print(img)
    img = img.transpose((1,2,0)) ## Converting image to (224,224,3) to do normalization with mean and std.
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    if normalize:
        img = ((img - mean) / std)
    img = img.transpose((2,0,1)) ## Converting image back to (3,224,224) which our model expects for precition. 
    img = torch.tensor(img,dtype=torch.float32) ## Converting back to pytorch tensor.
    return img


def predict(image_dir, filepath_pth, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():    
        img = process_image(image_dir)
        model, optimizer, loss, class_to_idx = load_checkpoint(filepath_pth)
        output = model.forward(img.unsqueeze(0).to(device) if len(img.size())==3 else img.to(device))
        top_5_probs,classes = output.topk(topk)
        return top_5_probs, classes


# Call to main function to run the program
if __name__ == "__main__":
    main()