import re
import sys
import getopt
import os
import csv
import color_moment_alternateV
from sklearn.preprocessing import StandardScaler

from numpy.linalg import eig
import features
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import image_to_db

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "q:l:n:") # Grabbing all of the available flags
    except getopt.GetoptError: # Throws an error for not having a correct flag, or a flag in the wrong position
        print('Usage: p2task5.py -q <query image file>, -l <latent semantics file> -n <number of similar images>')
        sys.exit(2) # Exit the program
    for opt, arg in opts: # For every flag and arg
        if opt in ("-q"): # If flag is -q
            query_file = arg # Store the query filename
        if opt in ("-l"): # If flag is -l
            latent_semantics_file = arg # Store the latent semantics filename
        if opt in ("-n"): # If flag is -n
            n_val = int(arg) # Store the number of similar images to find

    dir = "phase2_data/all/"

    # types in order for task 3 similarity matrix
    types = ['jitter', 'cc', 'con', 'emboss', 'neg', 'noise01', 'noise02', 'original', 'poster', 'rot', 'smooth', 'stipple']

    # reduction
    # feature
    # subject_or_type
    # k_ls

    file_split = re.split('_|\.', latent_semantics_file)
    if (file_split[0] == 'task12'):
        if (file_split[3].isnumeric()): # task 1
            subject_or_type = int(file_split[3])
        else: # task 2
            subject_or_type = file_split[3]
        k_ls = file_split[4]
    elif (file_split[0] == 'task3'):
        k_ls = file_split[3]
    else:
        print('Invalid latent semantics file, it should be from tasks 1-3')
        sys.exit(2) # Exit the program


    image = Image.open(dir + query_file) # Open the image
    img = np.asarray(image) # Make the image a numpy array
    if (file_split[2] == 'CM'):
        feature = color_moment_alternateV.combineColorMoments
    elif(file_split[2] == 'ELBP'):
        feature = features.histogramFeatureDescriptorELBP
    elif(file_split[2] == 'HOG'):
        feature = features.HOG
    model = file_split[2]
    if (file_split[1] == 'SVD'):
        reduction = features.svd
    elif(file_split[1] == 'PCA'):
        eigen = image_to_db.db_to_object('task12_' + 'PCA' + "_" + model + "_" + subject_or_type + "_" + k_ls + ".ei")['eigenpairs']
        print(eigen)
        for i in range(len(eigen)):
            eigen[i][1] = np.array(eigen[i][1])
        print(eigen)
        ag = 0
        for i in range(1):
            ag = ag + img
        fd = feature(ag)
        X_std = StandardScaler().fit_transform(fd.reshape(-1,1))
        for j in range(len(eigen)):
            print(X_std.T.dot(eigen[j][1]))
        sys.exit(2)
    
    fd = feature(img)

    i = 0
    imgs = []
    weights = []
    x_weight_pairs = []
    reader_bool = True
    with open(latent_semantics_file, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            print("row", row)
            if (row != ['ls' + str(i)]):
                if reader_bool:
                    imgs = row
                else:
                    weights = row
                    i = i + 1
                reader_bool = not reader_bool
            elif i > 0:
                x_weight_pairs.append(dict(zip(imgs, weights)))
    x_weight_pairs.append(dict(zip(imgs, weights)))
    print(x_weight_pairs)
    print(ls)

    # f, axarr = plt.subplots(n_val+1,gridspec_kw = {'wspace':0, 'hspace':0})
    # y = 0
    # first = True
    # # testing
    # img_filenames = os.listdir("phase2_data/all/")
    # img_filenames = img_filenames[:n_val]
    # print(img_filenames)
    # for img in img_filenames:
    #     img_decoded = Image.open(dir + img)
    #     axarr[y].imshow(img_decoded, cmap=plt.cm.gray)
    #     y += 1
    # plt.show()

    
if __name__ == "__main__": # Setup for main
    main(sys.argv[1:]) # Call main with all args