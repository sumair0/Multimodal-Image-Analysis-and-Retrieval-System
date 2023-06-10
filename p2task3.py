import sys
import getopt
import os
import numpy as numpy
import scipy as scipy
from scipy.stats import skew
import matplotlib.pyplot as plt
from skimage import io
import os
import skimage.feature
from color_moment_alternateV import combineColorMoments
import features
import csv

types = ['jitter', 'cc', 'con', 'emboss', 'neg', 'noise01', 'noise02', 'original', 'poster', 'rot', 'smooth', 'stipple']

def computeTypeTypeSimilarityMatrix(featureModel):
    #fetch images as per names ,combine and normalize

    jitterImages = numpy.zeros((64,64))
    ccImages = numpy.zeros((64,64))
    conImages = numpy.zeros((64,64))
    detailImages = numpy.zeros((64,64))
    embossImages = numpy.zeros((64,64))
    negImages = numpy.zeros((64,64))
    noise01Images = numpy.zeros((64,64))
    noise02Images = numpy.zeros((64,64)) 
    originalImages = numpy.zeros((64,64)) 
    posterImages = numpy.zeros((64,64))
    rotImages = numpy.zeros((64,64))
    smoothImages = numpy.zeros((64,64)) 
    stippleImages = numpy.zeros((64,64))  

    list_files = os.listdir('./all_images')

    for fileName in list_files:
        if 'jitter' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            jitterImages = jitterImages + Image
        if 'cc' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            ccImages = ccImages + Image
        if 'con' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            conImages = conImages + Image
        if 'emboss' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            embossImages = embossImages + Image
        if 'neg' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)

            negImages = negImages + Image
        
        if 'noise01' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            noise01Images = noise01Images + Image
        
        if 'noise02' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            noise02Images = noise02Images + Image
        
        if 'original' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)

            originalImages = originalImages + Image
        
        if 'poster' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            posterImages = posterImages + Image
        
        if 'rot' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            rotImages = rotImages + Image
        
        if 'smooth' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            smoothImages = smoothImages + Image
        
        if 'stipple' in fileName:
            Image = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
            Image = Image/numpy.amax(Image)
        
            stippleImages = stippleImages + Image

    if featureModel == "CM" :
        jitterFeatureRepresentation = combineColorMoments(jitterImages)
        ccFeatureRepresentation = combineColorMoments(ccImages)
        conFeatureRepresentation = combineColorMoments(conImages)
        embossFeatureRepresentation = combineColorMoments(embossImages)
        negFeatureRepresentation = combineColorMoments(negImages)
        noise01FeatureRepresentation = combineColorMoments(noise01Images)
        noise02FeatureRepresentation = combineColorMoments(noise02Images)
        originalFeatureRepresentation = combineColorMoments(originalImages)
        posterFeatureRepresentation = combineColorMoments(posterImages)
        rotFeatureRepresentation = combineColorMoments(rotImages)
        smoothFeatureRepresentation = combineColorMoments(smoothImages)
        stippleFeatureRepresentation = combineColorMoments(stippleImages)
        lengthOfFeatureModel = len(ccFeatureRepresentation)

        M_N_preprocess = numpy.concatenate((jitterFeatureRepresentation,ccFeatureRepresentation,conFeatureRepresentation,embossFeatureRepresentation,\
                          negFeatureRepresentation,noise01FeatureRepresentation,noise02FeatureRepresentation,\
                          originalFeatureRepresentation,posterFeatureRepresentation,rotFeatureRepresentation,\
                          smoothFeatureRepresentation,stippleFeatureRepresentation), axis = 0)

        M_N_representation = M_N_preprocess.reshape(-1,lengthOfFeatureModel)

    elif featureModel == "HOG" :
        jitterFeatureRepresentation = skimage.feature.hog(jitterImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        ccFeatureRepresentation = skimage.feature.hog(ccImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        conFeatureRepresentation = skimage.feature.hog(conImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        embossFeatureRepresentation = skimage.feature.hog(embossImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        negFeatureRepresentation = skimage.feature.hog(negImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        noise01FeatureRepresentation = skimage.feature.hog(noise01Images, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        noise02FeatureRepresentation = skimage.feature.hog(noise02Images, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        originalFeatureRepresentation = skimage.feature.hog(originalImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        posterFeatureRepresentation = skimage.feature.hog(posterImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        rotFeatureRepresentation = skimage.feature.hog(rotImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        smoothFeatureRepresentation = skimage.feature.hog(smoothImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        stippleFeatureRepresentation = skimage.feature.hog(stippleImages, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        lengthOfFeatureModel = len(ccFeatureRepresentation)

        M_N_preprocess = numpy.concatenate((jitterFeatureRepresentation,ccFeatureRepresentation,conFeatureRepresentation,embossFeatureRepresentation,\
                          negFeatureRepresentation,noise01FeatureRepresentation,noise02FeatureRepresentation,\
                          originalFeatureRepresentation,posterFeatureRepresentation,rotFeatureRepresentation,\
                          smoothFeatureRepresentation,stippleFeatureRepresentation), axis = 0)

        M_N_representation = M_N_preprocess.reshape(-1,lengthOfFeatureModel)

    elif featureModel == "ELBP":
        jitterFeatureRepresentation = features.histogramFeatureDescriptorELBP(jitterImages)
        ccFeatureRepresentation = features.histogramFeatureDescriptorELBP(ccImages)
        conFeatureRepresentation = features.histogramFeatureDescriptorELBP(conImages)
        embossFeatureRepresentation = features.histogramFeatureDescriptorELBP(embossImages)
        negFeatureRepresentation = features.histogramFeatureDescriptorELBP(negImages)
        noise01FeatureRepresentation = features.histogramFeatureDescriptorELBP(noise01Images)
        noise02FeatureRepresentation = features.histogramFeatureDescriptorELBP(noise02Images)
        originalFeatureRepresentation = features.histogramFeatureDescriptorELBP(originalImages)
        posterFeatureRepresentation = features.histogramFeatureDescriptorELBP(posterImages)
        rotFeatureRepresentation = features.histogramFeatureDescriptorELBP(rotImages)
        smoothFeatureRepresentation = features.histogramFeatureDescriptorELBP(smoothImages)
        stippleFeatureRepresentation = features.histogramFeatureDescriptorELBP(stippleImages)
        lengthOfFeatureModel = len(ccFeatureRepresentation)

        M_N_preprocess = numpy.concatenate((jitterFeatureRepresentation,ccFeatureRepresentation,conFeatureRepresentation,embossFeatureRepresentation,\
                          negFeatureRepresentation,noise01FeatureRepresentation,noise02FeatureRepresentation,\
                          originalFeatureRepresentation,posterFeatureRepresentation,rotFeatureRepresentation,\
                          smoothFeatureRepresentation,stippleFeatureRepresentation), axis = 0)

        M_N_representation = M_N_preprocess.reshape(-1,lengthOfFeatureModel)
        

    else:
        print("Please specify correct featuremodel , reductiontechnique and k")



    typeTypeSimilarityMatrix = scipy.spatial.distance.cdist(M_N_representation,M_N_representation, metric='cosine')

    return typeTypeSimilarityMatrix


def reportTopKLatentSemantics(TTSimilarityMatrix, reductionTechnique, k):
    '''
    input : reductionTechnique : {PCA, SVD} , k
    featureModel : CM, ELBP, HOG
    returts : topKLatentSemantics
    '''
    topKLatentSemantics = []

    if reductionTechnique == 'PCA':
        topKLatentSemantics = features.pca_on_square_matrix(TTSimilarityMatrix,k)

    # elif reductionTechnique == 'SVD':
    #     topKLatentSemantics = features.svd()

    else:
        print("Please specify the correct reduction technique and K")

    
    return topKLatentSemantics



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:k:d:") # Grabbing all of the available flags
    except getopt.GetoptError: # Throws an error for not having a correct flag, or a flag in the wrong position
        print('Usage: p2task3.py -m <CM/ELBP/HOG>, -k <val> -d <PCA/SVD>')
        sys.exit(2) # Exit the program
    for opt, arg in opts: # For every flag and arg
        if opt in ("-m"): # If flag is -m
            model = arg # Store the model type
        if opt in ("-k"): # If flag is -k
            k_val = arg # Store the k-value
        if opt in ("-d"): # If flag is -d
            dim_red = arg # Store the dimension reduction technique

    out_file = dim_red + '_' + model + '_' + k_val + '.csv'
    typetype = computeTypeTypeSimilarityMatrix(model)
    topk = reportTopKLatentSemantics(typetype, dim_red, int(k_val))
    print(topk)
    print(typetype)
    with open('task3_similarity_' + out_file, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        #writer.writerow(types)
        for i in range(12):
            writer.writerow(typetype[i])
    
    with open('task3_' + out_file, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        for i in range(int(k_val)):
            writer.writerow(["ls" + str(topk[i][0])])
            writer.writerow(topk[i][1].keys())
            writer.writerow(topk[i][1].values())
    

if __name__ == "__main__": # Setup for main
    main(sys.argv[1:]) # Call main with all args