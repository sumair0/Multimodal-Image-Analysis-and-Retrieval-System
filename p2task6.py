import io
import sys
import getopt
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "q:l:n:") # Grabbing all of the available flags
    except getopt.GetoptError: # Throws an error for not having a correct flag, or a flag in the wrong position
        print('Usage: p2task5.py -q <query image file>, -l <latent semantics file>')
        sys.exit(2) # Exit the program
    for opt, arg in opts: # For every flag and arg
        if opt in ("-q"): # If flag is -q
            query_file = arg # Store the query filename
        if opt in ("-l"): # If flag is -l
            latent_semantics_file = arg # Store the latent semantics filename

    dir = "phase2_data/all/"

    
    
if __name__ == "__main__": # Setup for main
    main(sys.argv[1:]) # Call main with all args