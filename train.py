# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#                                                                             
# PROGRAMMER: Chung Leuk Lee.
# DATE CREATED: 06/09/2018                                  
# REVISED DATE: 06/23/2018  
# PURPOSE: This script trains a neural network based on transfer characteristics
#          of a pre-trained neural network(e.g, vgg19),
#          The model will be trained to predict the category of
#          the input image. then saves the newly trained
#          model. 
#
# Use argparse Expected Call with <> indicating expected user input:
#      e.g. python train.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python predict.py --data_dir pet_images/ --arch vgg19 
##

import argparse
from time import time, sleep
from Image_Classifier import *

def main():
    """
    Main function for train.py - parses command line arguments for data directory,
    save directory, checkpoint name, transfer model architecture, learning rate,
    classifier network hidden units dimensions, number of epochs, dropout probability,
    and whether or not to use the GPU for training. 
    """
    # Measures total program runtime by collecting start time
    start_time = time.time()
    
    #Creates & retrieves Command Line Arugments
    in_arg = get_input_args()

    # Set the default hyperparameters if none given
    if in_arg.hidden_units==[]:
        in_arg.hidden_units=[8192, 4096, 2048, 1024];

    # Create the save file path name
    save_path = in_arg.checkpoint

    # Organize the inputs into the hyperparameters dictionary
    hyperparameters = {'architecture': in_arg.arch,
                       'hidden_layers': in_arg.hidden_units,
                       'dropout_rate': in_arg.drop_rate,
                       'learn_rate': in_arg.learning_rate,
                       'epochs': in_arg.epochs}

    # Get the transfer model
    model = get_model(hyperparameters['architecture'])

    # Create the dataloaders for training and testing images
    # Also, create the classifier based on the given inputs and attach it to the transfer model
    model, trainloader, testloader, validloader = model_config(in_arg.data_dir, model, 
                                                                         hyperparameters['hidden_layers'], 
                                                                         hyperparameters['dropout_rate'])

    # Print the relevant parameters to the output window before training
    print("\n")
    print("Transfer model:               {}".format(hyperparameters['architecture']))
    print("Hidden layers:                {}".format(hyperparameters['hidden_layers']))
    print("Learning rate:                {}".format(hyperparameters['learn_rate']))
    print("Dropout:                      {}".format(hyperparameters['dropout_rate']))
    print("Epochs:                       {}".format(hyperparameters['epochs']))
    print("Training data:                {}".format(in_arg.data_dir + '/train'))
    print("Validation data:              {}".format(in_arg.data_dir + '/test'))
    print("Checkpoint will be saved as:  {}".format(save_path))

    # Start training the network and return the model accuracy after training
    model_accuracy = network_training(model, 
                                   hyperparameters['epochs'], 
                                   hyperparameters['learn_rate'], 
                                   trainloader, testloader, 
                                   in_arg.gpu)

    # Save the model checkpoint along with the transfer model architecture it came from and the model's trained accuracy
    save_checkpoint(save_path, model, hyperparameters['architecture'], 
                    model.classifier.hidden_layers[0].in_features, 
                    model.classifier.output.out_features, 
                    hyperparameters, model_accuracy)
    #Measure total program runtime by collecting end time
    end_time = time.time()

    #Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", str(int((tot_time / 3600))) + ":" + 
          str(int( (tot_time % 3600) / 60)) + ":" + str(int((tot_time % 3600) % 60)))

    
def get_input_args():
    """
    Parse command line arguments.
    usage: train.py [-h] [--save_dir SAVE_DIR] [--checkpoint CHECKPOINT]
                [--arch ARCH] [--learning_rate LEARNING_RATE]
                [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS]
                [--drop_p DROP_P] [--gpu]
                data_dir
    positional arguments:
      data_dir              full path name to data directory of categorized
                            images; must contain folders /train and /test;
                            example: flowers
    optional arguments:
      -h, --help            show this help message and exit
      --save_dir SAVE_DIR   full path directory to save model checkpoints;
                            Default: model_checkpoints/
      --checkpoint CHECKPOINT
                            name of checkpoint file to save (name only, not path);
                            Default: checkpoint.pth
      --arch ARCH           chosen model; Default: vgg19; choices: vgg16, vgg13,
                            vgg11, alexnet
      --learning_rate LEARNING_RATE
                            learning rate; float; Default: 0.0001
      --hidden_units HIDDEN_UNITS
                            append a hidden layer unit - call multiple times to
                            add more layers; integer; example: --hidden_units 128 --hidden_units 64
      --epochs EPOCHS       number of epochs; integer number only; Default - 16
      --drop_out            dropout probability; float; Default - 0.2
      --gpu                 use GPU instead of CPU; Default - False
    Parameters:
        None
    Returns:
        parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', action="store", type=str,default='flowers/train',
                        help='full path name to data directory of categorized images; example: flowers/train')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help='name of checkpoint file to save (name only, not path); Default: checkpoint.pth')
    parser.add_argument('--arch', type=str, default='vgg19',
                        help='chosen model; Default: vgg19; choices: vgg19, vgg16, vgg13, alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate; float; Default: 0.0001')
    parser.add_argument('--hidden_units', action="append", type=int, default=[],
                        help='append a hidden layer unit - call multiple times to add more layers; integer; default:[8192, 4096, 2048, 1024]')
    parser.add_argument('--epochs', type=int, default=16,
                        help='number of epochs; integer number only; Default - 16')
    parser.add_argument('--drop_rate', type=float, default=0.2,
                        help='dropout probability; float; Default - 0.2')
    parser.add_argument('--gpu', action="store_true", default=True,
                        help='use GPU instead of CPU; Default - True')
    return parser.parse_args()


if __name__ == '__main__':
    main()