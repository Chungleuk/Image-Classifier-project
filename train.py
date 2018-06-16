# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Chung Leuk Lee.
# DATE CREATED: 06/09/2018                                  
# REVISED DATE: 06/15/2018  
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python predict.py --dir pet_images/ --arch vgg 
##

# Imports here
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
    #Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
     
    network(in_arg.arch, in_arg.lr, in_arg.epoch)

    #Measure total program runtime by collecting end time
    end_time = time()

    #Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", str(int((tot_time / 3600))) + ":" + 
          str(int( (tot_time % 3600) / 60)) + ":" + str(int((tot_time % 3600) % 60)))

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
    # Create command line arguments 
    parser.add_argument('--arch', type=str, default='densenet121', help='chosen model')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=24, help='traning epochs')
    # returns parsed argument collection
    return parser.parse_args()

#data path
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Load the data
# Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

vali_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=vali_transforms)
test_data = datasets.ImageFolder(test_dir, transform=vali_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

#Label
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Building and training the classifier
## We check whether GPU is available on PC or not otherwise we use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

##Model building
vgg16 = models.vgg16(pretrained=True).to(device)
resnet18 = models.resnet18(pretrained=True).to(device)
densenet121 = models.densenet121(pretrained=True).to(device)

# TODO: Build and train your network

def network(model_arch,lr_lr,epochs_epoch):
    if model_arch == 'vgg':
       # Freeze parameters
        for param in vgg16.parameters():
            param.requires_grad = False   

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
        vgg16.classifier = classifier
        vgg16.classifier = vgg16.classifier.to(device)
        for param in vgg16.classifier.parameters():
            param.requires_grad = True

        # training network
        start_time = time()
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(vgg16.classifier.parameters(), lr_lr, momentum=0.9)

        print_every = 40
        steps = 0
        vgg16.train()   
        for e in range(epochs_epoch):
            running_loss = 0
            t = 0
            for images, labels in trainloader:
                steps += 1
                #input data
                images, labels = Variable(images), Variable(labels)
                images, labels = images.to(device), labels.to(device)  ## We call to(device) to transfer tensors between CPU and GPU
                #clear gradients calculated in previous step
                optimizer.zero_grad()
                #forward+backward
                outputs = vgg16.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                #renew parameter updated according to new gradients and learning rate.
                optimizer.step()
                _, preds = torch.max(outputs.data, 1)
                ## We caculate from this particular batch how many were caculated/guessed correctly by model.
                correct += (labels == preds).sum()
                #print out
                running_loss += loss.item()
                ## We print loss at every 40 batches to check whether it's decresing or not. Ideally it should decrease.
                if steps % print_every == 0:
                    print("Epoch: {}/{}.. ".format(e + 1, epochs_epoch), "Training Loss: {:4f}".format(running_loss / print_every))
                    running_loss = 0
            print('Accuracy at %d epoch & Learning Rate : %f is %f %%' % (e + 1, lr_lr, float(correct) * 100 / len(trainloader.dataset)))
            end_time = time()
        print('Finished Training in time : %f seconds' % (end_time  - start_time))

        #Testing  network
        vgg16.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in validloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = vgg16(Variable(images))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Accuracy of the network on the %d valid images: %d %%' % (len(validloader.dataset), 100 * correct / total))

        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = vgg16(Variable(images))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Accuracy of the network on the %d test images: %d %%' % (len(testloader.dataset), 100 * correct / total))

        vgg16.class_to_idx = train_data.class_to_idx
        model_state_dict = vgg16.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        loss_state_dict = criterion.state_dict()

        checkpoint_dict = {
            'epochs': epochs_epoch,
            'model_dict': model_state_dict,
            'optimizer': optimizer_state_dict,
            'loss': loss_state_dict,
            'class_to_idx': train_data.class_to_idx,
             }
        torch.save(checkpoint_dict, 'vgg.pth')
    else:
        for param in densenet121.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features=1024, out_features=512)),
                          ('dropout1',nn.Dropout(0.5)),
                          ('relu1', nn.ReLU()), 
                          ('fc2', nn.Linear(in_features=512, out_features=102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        densenet121.classifier = classifier
        densenet121.classifier = densenet121.classifier.to(device)
        for param in densenet121.classifier.parameters():
            param.requires_grad = True
        
        start_time = time()
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(densenet121.classifier.parameters(), lr_lr, momentum=0.9)

        print_every = 40
        steps = 0
        densenet121.train()
      
        for e in range(epochs_epoch):
            running_loss = 0
            correct = 0
            for images, labels in trainloader:
                steps += 1
                #input data
                images, labels = Variable(images), Variable(labels)
                images, labels = images.to(device), labels.to(device) 
                #clear gradients calculated in previous step
                optimizer.zero_grad()
                #forward +backward
                outputs = densenet121.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                #renew parameter updated according to new gradients and learning rate.
                optimizer.step()
                _, preds = torch.max(outputs.data, 1)
                ## We caculate from this particular batch how many were caculated/guessed correctly by model.
                correct += (labels == preds).sum()
                #print out
                running_loss += loss.item()
                ## We print loss at every 40 batches to check whether it's decresing or not. Ideally it should decrease.
                if steps % print_every == 0:
                    print("Epoch: {}/{}.. ".format(e + 1, epochs_epoch), "Training Loss: {:4f}".format(running_loss / print_every))
                    running_loss = 0
            print('Accuracy at %d epoch & Learning Rate : %f is %f %%' % (e + 1, lr_lr, float(correct) * 100 / len(trainloader.dataset)))
            end_time = time()
        print('Finished Training in time : %f seconds' % (end_time  - start_time))

        #Testing  network
        densenet121.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in validloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = densenet121(Variable(images))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Accuracy of the network on the %d valid images: %d %%' % (len(validloader.dataset), 100 * correct / total))

        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = densenet121(Variable(images))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Accuracy of the network on the %d test images: %d %%' % (len(testloader.dataset), 100 * correct / total))

        densenet121.class_to_idx = train_data.class_to_idx
        model_state_dict = densenet121.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        loss_state_dict = criterion.state_dict()

        checkpoint_dict = {
            'epochs': epochs_epoch,
            'model_dict': model_state_dict,
            'optimizer': optimizer_state_dict,
            'loss': loss_state_dict,
            'class_to_idx': train_data.class_to_idx,
             }
        torch.save(checkpoint_dict, 'densenet.pth')


# Call to main function to run the program
if __name__ == "__main__":
    main()

