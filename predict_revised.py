#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Chung Leuk Lee.
# DATE CREATED: 06/09/2018                                  
# REVISED DATE: 06/28/2018  #modify and test4
# PURPOSE: 
#          This script Check images & report results: 
#          read them in, predict their content (classifier), 
#          compare prediction to actual value labels
#          and output results.   
#
#    Example call:
#    python predict.py --pth vgg.pth --dir flowers/valid/1/image_06739.jpg 
##

import argparse
from PIL import Image
import time
import json
from Image_Classifier import *

def main():
    """
    Main function for predict.py - Loads a model checkpoint from a saved model.
    Loads the category class names from the *.json file. 
    Predicts the output along with top k probabilities.
    
    """
    #Measures total program runtime by collecting start time
    start_time = time.time()
    
    #Creates & retrieves Command Line Arugments
    in_arg = get_input_args()

    # Load the model checkpoint
    print("\nLoading model checkpoint: {}\n".format(in_arg.checkpoint))
    model, accuracy = load_checkpoint(in_arg.checkpoint)

    # Get the category names mapping if the file was provided
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    model.class_to_idx = cat_to_name
        
    if in_arg.gpu==True:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU not available. Using CPU instead.")
            device = torch.device("cpu")
    # Predict the probabilities and top k classes
    print("Predicting image category...")
    probs_tensor, classes_tensor = predict(in_arg.input, model.to(device), in_arg.top_k)

    # If the category mapping file was provided, make a list of names
    if in_arg.category_names!='':
        classes = [model.class_to_idx[str(sorted(model.class_to_idx)[i])] for i in (classes_tensor).tolist()[0]]
    # Otherwise create a list of index numbers
    else:
        classes = classes_tensor.tolist()[0]
    # Convert the probabilities and classes tensors into lists
    probs = probs_tensor.data[0].cpu().numpy()
    probs = np.exp(probs)
	# Print the predicted category to output
    print("Given image: {}".format(in_arg.input))
    print("Predicted category: {}".format(classes))
    print("Predicted probability: {} % ".format(probs*100))
  
    #Measure total program runtime by collecting end time
    end_time = time.time()

    #Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", str(int((tot_time / 3600))) + ":" + 
          str(int( (tot_time % 3600) / 60)) + ":" + str(int((tot_time % 3600) % 60)))
       

def get_input_args():
    """
    Parse command line arguments.
    usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu] [--input_category INPUT_CATEGORY]
                  input checkpoint
    positional arguments:
      input                 full path name to input image file; example:
                            flowers/valid/63/image_05876.jpg; 
      checkpoint            the full path to a checkpoint file *.pth of a trained
                            network; example: model_checkpoints/checkpoint.pth
    optional arguments:
      -h, --help            show this help message and exit
      --top_k TOP_K         top k results from classifier; integer number
      --category_names CATEGORY_NAMES
                            the full path to a *.json file mapping categories to
                            names; example: cat_to_name.json   
    Parameters:
        None
    Returns:
        parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action="store", type=str, default='/home/workspace/aipnd-project/flowers/valid/1/image_06749.jpg',
                        help='full path name to input image file; example: /home/workspace/aipnd-project/flowers/valid/1/image_06739.jpg')
    parser.add_argument('--checkpoint', action="store", type=str, default='checkpoint.pth',
                        help='the full path to a checkpoint file *.pth of a trained network; example: checkpoint.pth')
    parser.add_argument('--top_k', type=int, default=5,
                        help='top k results from classifier; integer number')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='the full path to a *.json file mapping categories to names; example: cat_to_name.json')
    parser.add_argument('--gpu', action="store", default=True,
                        help='pass this argument to use GPU, else use CPU; default: True' )
    
    return parser.parse_args()


if __name__ == '__main__':
    main()