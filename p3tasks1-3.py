import sys
import getopt
import os
import scipy
import csv
from PIL import Image

import ssl
from connection_string import connection_string
from pymongo import MongoClient
from image_to_db import db_to_object, image_to_db
from encode_image import decode_image

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:x:k:d:") # Grabbing all of the available flags
    except getopt.GetoptError: # Throws an error for not having a correct flag, or a flag in the wrong position
        print('Usage: p3tasks1-3.py -t <1/2/3>, -m <CM/ELBP/HOG>, -k <val>, -i <img_path> ')
        sys.exit(2) # Exit the program
    for opt, arg in opts: # For every flag and arg
        if opt in ("-t"): # If flag is -t
            task = arg # Store the task
        if opt in ("-m"): # If flag is -m
            model = arg # Store the model type
        if opt in ("-k"): # If flag is -k
            k_val = arg # Store the k-value
        if opt in ("-i"): # If flag is -d
            img_path = arg # Store the image folder path
    
    # extract features,
    # compute latent semantics,
    # classifiers etc.

if __name__ == "__main__": # Setup for main
    main(sys.argv[1:]) # Call main with all args