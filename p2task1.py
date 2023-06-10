import sys
import getopt
import os

import scipy
import csv
from numpy.linalg import eig
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import numpy
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation

import features

import ssl
from connection_string import connection_string
from pymongo import MongoClient
from image_to_db import db_to_object, image_to_db
from encode_image import decode_image

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:x:k:d:") # Grabbing all of the available flags
    except getopt.GetoptError: # Throws an error for not having a correct flag, or a flag in the wrong position
        print('Usage: p2task1.py -m <CM/ELBP/HOG>, -x <{cc/con/detail/emboss/jitter/neg/noise1/noise2/original/poster/rot/smooth/stipple> -k <val> -d <PCA/SVD>')
        sys.exit(2) # Exit the program
    for opt, arg in opts: # For every flag and arg
        if opt in ("-m"): # If flag is -m
            model = arg # Store the model type
        if opt in ("-x"): # If flag is -x
            x_val = arg # Store the x-value
        if opt in ("-k"): # If flag is -k
            k_val = arg # Store the k-value
        if opt in ("-d"): # If flag is -d
            dim_red = arg # Store the dimension reduction technique

    out_file = 'task12_' + dim_red + "_" + model + "_" + x_val + "_" + k_val + ".csv"
    out_eigen = 'task12_' + dim_red + "_" + model + "_" + x_val + "_" + k_val + ".ei"

    # Database connection
    client = MongoClient(connection_string(), ssl_cert_reqs=ssl.CERT_NONE)
    db = client.cse515

    subject_weight_pairs = [] # List to store the subject-weight pairs
    subject_weight_pairs = features.pca_task1(x_val, model, int(k_val), out_eigen)
    # for i in range(int(k_val)):
    #     subject_weight_pairs.append({})
    # count = 0 # * Incrementer for testing
    # for file in os.listdir("phase2_data/all/"): # For every file in the image directory NOTE: May need to change the directory location depending on your computer
    #     if x_val in file: # If the image type name is in the filename
    #         image = Image.open("phase2_data/all/" + file) # Open the image
    #         img = numpy.asarray(image) # Make the image a numpy array

    #         # get feature from database or calculate here
    #         im = db.phase2.find_one({'id': file})

    #         #print(hasattr(im, model), hasattr(model, dim_red))

    #         # change this if you want to force upload even if in database already
    #         # the use case being that the feature algorithm changes
    #         force_upload = False

    #         if (im != None and hasattr(im, model) and hasattr(model, dim_red) and not force_upload):
    #             ls = im.dim_red
    #         else:
    #             if model == 'CM': # If the specified model is Color Moments
    #                 fd = features.color_moments(img) # Obtain the Color Moments of the current image
    #                 #fd = []
    #                 #for i in range(3):
    #                 #    sum = 0
    #                 #    for j in range(64):
    #                 #        sum += temp[j][i]
    #                 #    fd.insert(len(fd), sum/64)
    #                 #if count == 0:
    #                 #    print(fd)
                    
    #             elif model == 'ELBP': # If the specified model is ELBP
    #                 fd = features.ELBP(img) # Obtain the ELBP of the current image

    #             else: # If the specified model is HOG
    #                 fd = features.HOG(img) # Obtain the HOG of the current image

    #             if dim_red == "PCA": # If the dimensionality reduction technique is PCA
    #                 ls = features.pca(image, model, int(k_val))

    #             elif dim_red == "SVD": # If the dimensionality reduction technique is SVD
    #                     # print("fd len", len(fd))
    #                 ls = features.svd(fd, model, int(k_val), False)
    #                 # U, S, VT = numpy.linalg.svd(fd) # ! Cant use this function according to Piazza
    #                 # if count == 0: # * For testing purposes
    #                 #     print(U.shape)
    #                 #     print(S.shape)
    #                 #     print(VT.shape)

    #             elif dim_red == "LDA": # If the dimensionality reduction technique is LDA
    #                 features.lda(fd, model, int(k_val))

    #             else: # If the dimensionality reduction technique is K-means
    #                 x=1

    #             # this if statement is temporary until all reduction techniques are done
    #             if (ls):
    #                 image_to_db("phase2_data/all/", file, model, fd.tolist(), dim_red, ls)
                
    #             for i in range(int(k_val)):
    #                 subject_weight_pairs[i][file] = ls[i]

    #     count += 1 # Increment counter

    # print("len", len(subject_weight_pairs[0]))
    # for i in range(int(k_val)):
    #     subject_weight_pairs[i] = dict(sorted(subject_weight_pairs[i].items(), key=lambda item: item[1], reverse=True))
    print(subject_weight_pairs)
    with open(out_file, 'w', newline='') as output_file:
        for i in range(int(k_val)):
            writer = csv.writer(output_file)
            writer.writerow(["ls" + str(subject_weight_pairs[i][0])])
            writer.writerow(subject_weight_pairs[i][1].keys())
            writer.writerow(subject_weight_pairs[i][1].values())

if __name__ == "__main__": # Setup for main
    main(sys.argv[1:]) # Call main with all args