# coding: utf-8
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Chung Leuk Lee.
# DATE CREATED: 06/09/2018                                  
# REVISED DATE: 06/15/2018  #modify and test2

#This module contains functions and classes relating to the neural network classifier.
#This module is referenced by train.py and predict.py
#Classes:
#    Network(nn.Module) - Builds a feedforward network with arbitrary hidden layers
#Functions:
#    get_model() - Parse the given model name and create a new transfer model
#    model_config() - 
#    validation() - 
#    network_training() - 
#    save_checkpoint() - 
#    load_checkpoint() - 
#    process_image() - 
#    predict() - 



#Imports here
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable

def main():
    print("This script contains functions and classes relating to train.py and predict.py")


def model_config(data_dir, model, hidden_layers, dropout_rate):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([ transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    vali_transforms = transforms.Compose([ transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([ transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=vali_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=vali_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
     # Check the classifier of the given model and find the first Linear module to see how many inputs it takes
    classifier_modules = [type(model.classifier[i]).__name__ for i in range(len(model.classifier))]
    First_Linear_Module = classifier_modules.index('Linear')
     
    # Create the classifier for the model 
    # Use the transfer models outputs as inputs and the number of image categories as outputs

    classifier = Network(model.classifier[First_Linear_Module].in_features, len(train_data.classes), hidden_layers, dropout=dropout_rate)
    # Attach the classifier to the model
    model.classifier = classifier
    return model, trainloader, validloader, testloader,


def validation(model, testloader, criterion, processor):
    """
    Validates the model using the test images.
    Parameters:
        model - torchvision.models.model, the neural network model
        testloader - the dataloader for the test set
        criterion - the output loss criterion
        processor - torch.device, CPU of GPU
    Returns:
        test_loss - torch.tensor, the output loss from the validation pass
        accuracy - torch.tensor, the percentage of correctly classified test images
    """
    test_loss = 0
    accuracy = 0

    # Send the model to the desired processor
    model.to(processor)

    # Iterate through the test dataloader and do a validation pass on all images
    for images, labels in testloader:
        images, labels = images.to(processor), labels.to(processor)
        # Do a forward pass
        output = model.forward(images)

        # Calculate the loss based on the criterion
        test_loss += criterion(output, labels).item()

        # Since the output is log-loss, the probabilities are the e^x of the output
        ps = torch.exp(output)

        # Check how many of the most probable output results match the labels
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()   
    return test_loss, accuracy


def get_model(model_name):
    """
    Parse the given model name and create a new transfer model.
    Parameters:
        model_name - string, name of the transfer model from torchvision.models
    Returns:
        model - torchvision.models.model, the transfer network model with gradients disabled
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name=='vgg19':
        model = models.vgg19(pretrained=True).to(device)
    elif model_name=='vgg16':
        model = models.vgg16(pretrained=True).to(device)
    elif model_name=='vgg13':
        model = models.vgg13(pretrained=True).to(device)
    elif model_name=='alexnet':
        model = models.alexnet(pretrained=True).to(device)
    else:
        print("\nINPUT ERROR: must select from one of these models: vgg19, vgg16, vgg13, alexnet\n")
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    return model


#Defining class for a fully-connected network
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            dropout: float between 0 and 1
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)   
        return F.log_softmax(x, dim=1)


# training network
def network_training(model, epochs, learnrate, trainloader, testloader, GPU):

    # Select the GPU if it is available and if GPU is True
    if(GPU==True):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU not available. Using CPU instead.")
            device = torch.device("cpu")
    # Otherwise use the CPU
    else:
        device = torch.device("cpu")
    print(f"Device = {device}\n")

    # Set up the training parameters
    steps = 0
    running_loss = 0
    print_every = 40
    # Define the criterion and optimizer given the model and learning rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)

    # Send the model to the correct processor
    model.to(device)

    # Record the start time before training
    start = time.time()
    for e in range(epochs):
        running_loss = 0
        correct = 0
        for images, labels in trainloader:
            steps += 1
            #input data
            images, labels = Variable(images), Variable(labels)   
            images, labels = images.to(device), labels.to(device) ## We call to(device) to transfer tensors between CPU and GPU
            #clear gradients calculated in previous step
            optimizer.zero_grad()       
            #forward +backward
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()    
            #renew parameter updated according to new gradients and learning rate.
            optimizer.step()  
            _,preds = torch.max(outputs.data, 1)
            ## We caculate from this particular batch how many were caculated/guessed correctly by model.
            correct += (labels == preds).sum()

            #print out 
            running_loss += loss.item()
            
            ## We print loss at every 40 batches to check whether it's decresing or not. Ideally it should decrease.
            if steps % print_every == 0:
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:4f}".format(running_loss/print_every))
                running_loss = 0
        print('Accuracy at %d epoch & Learning Rate : %f is %f %%' % (e+1, learnrate,float(correct) *100 / len(trainloader.dataset)))

    print('Finished Training in time : %f seconds' % (time.time() - start))



def save_checkpoint(filepath, model, model_name, input_size, output_size, hyperparams, model_accuracy):
    """
    Save a model checkpoint.
    Parameters:
        filepath - string, the filepath of the checkpoint to be saved
        model - torchvision.models.model, the trained model to be saved
        model_name - string, the name of the transfer model 
        input_size - int, the input size of the classifier's hidden layers
        output_size - int, the output size of the classifier (number of categories)
        hyperparams - dictionary, the hyperparameters of the model training, including epochs, dropout probability, and learnrate
        model_accuracy - float, the accuracy of the trained model
    Returns:
        None
    """
    checkpoint = {'model_name': model_name,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hyperparams['hidden_layers'],
                  'dropout_rate': hyperparams['dropout_rate'],
                  'learn_rate': hyperparams['learn_rate'],
                  'epochs': hyperparams['epochs'],
                  'model_accuracy': model_accuracy,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath):
    """
    Load a model checkpoint.

    Parameters:
        filepath - string, the path to the saved checkpoint to be loaded
    Returns:
        model - the trained model from the checkpoint file
        accuracy - float, the accuracy of the trained model
    """
    checkpoint = torch.load(filepath)
    model_name = checkpoint['model_name']
    
    model = get_model(model_name)
    
    classifier = Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'],
                         checkpoint['dropout_rate'])
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    hidden_layers = checkpoint['hidden_layers']
    dropout_probability = checkpoint['dropout_rate']
    learnrate = checkpoint['learn_rate']
    epochs = checkpoint['epochs']
    accuracy = checkpoint['model_accuracy']   
    return model, accuracy


def process_image(image,normalize=True):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
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
    #print(img.shape)
    img = torch.tensor(img,dtype=torch.float32) ## Converting back to pytorch tensor.
    return img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    img = image.numpy()
    img = img.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    img = np.clip(img, 0, 1)  
    ax.imshow(img) 
    return ax

 
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        #img = Image.open(image_path)
        #img = test_transforms(img)
        img = process_image(image_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        #print(len(img.size()))
        ## Our model expects first dimension as batch_size hence using unsqueeze method 
        ## to translate (3,224,224) to (1,3,224,224)
        output = model.forward(img.unsqueeze(0).to(device) if len(img.size())==3 else img.to(device))
        top_5_probs,classes = output.topk(topk)
        return top_5_probs, classes

if __name__ == '__main__':
    main()

