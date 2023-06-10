import types
import numpy as np
import numpy
import pandas as pd
from sklearn import decomposition
from sklearn.datasets import fetch_olivetti_faces
from skimage.util.shape import view_as_windows
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.feature
from skimage import io
import os
import image_to_db

import  color_moment_alternateV
#standardization inbuilt library
from sklearn.preprocessing import StandardScaler


# Getting Olivetti Set
data = fetch_olivetti_faces()

# Storing data as ndarray
images = data.images
target = data.target

def color_moments(imageID):
    window_shape = (8, 8)
    row = 0
    col = 0
    fd = np.empty((0, 3), float)

    # Splits the picture into 8x8 windows for processing
    split = view_as_windows(imageID, window_shape)

    for row in range (0, 64, 8):
        for col in range (0, 64, 8):
            sum = np.sum(split[row,col])
            
            # Mean
            mean = sum/64

            # Standard Deviation
            sd = np.sqrt(np.sum(np.square(np.subtract(split[row, col], mean)))/64)

            # Skew
            skew = np.cbrt(np.sum(np.power(np.subtract(split[row, col], mean), 3))/64)

            descriptor = np.array([[mean, sd, skew]])
            fd = np.concatenate((fd, descriptor), axis=0)

    return fd

def ELBP(imageID):
    radius = 3
    n_points = 8*radius

    # Extended Local Binary Pattern - Grayscale Rotational Invariance
    elbp = local_binary_pattern(imageID, n_points, radius, method='ror')

    return elbp

# Histogram of Gradients
def HOG(imageID):
    feature_descriptors, hog_image = hog(imageID, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys', feature_vector=True)

    return feature_descriptors

def process(filename, name, value) -> None:
    image = mpimg.imread(filename, 0)
    plt.gray()
    plt.imshow(image)
    plt.title(value,fontsize=16)
    plt.suptitle(name,fontsize=24, y=1)
    plt.show()

def svd(fd, model, k, log=False):

    if model == 'HOG':
        fd = fd.reshape(1, -1)
    
    lam, v = LA.eig(fd.T @ fd)

    V = v[:, lam.argsort()[::-1]]

    lam_sorted = np.sort(lam)[::-1]
    lam_sorted = lam_sorted[lam_sorted > 1e-8]
    sigma = np.sqrt(lam_sorted)
    Sigma = np.zeros((fd.shape[0], fd.shape[1]))
    Sigma[:min(fd.shape[0],fd.shape[1]), :min(fd.shape[0],fd.shape[1])] = np.diag(sigma)

    r = len(sigma)
    U = fd @ V[:,:r] / sigma
    
    if log:
        print("U=", np.round(U, 4))
        print("Sigma=", np.round(Sigma, 4))
        print("V=", np.round(V, 4))

    topKLatentSemantics = []

    assert( k < len(Sigma[0]))

    for index in range(k):
        topKLatentSemantics.append(Sigma[index][index])
    
    return topKLatentSemantics
    

def lda(fd, model, k):
    if model == "CM":
        x=1 #standardize so no negative values

    elif model == 'HOG':
        fd = fd.reshape(1, -1)

    lda = decomposition.LatentDirichletAllocation(n_components=k, max_iter=100, random_state=0)
    lda.fit(fd)
    print(lda.transform(fd[-2:]))

def histogramFeatureDescriptorELBP(queryImage):
    X = skimage.feature.local_binary_pattern(queryImage, 8, 1, method='ror')/255
    histo_gram, bins = np.histogram(X)

    return histo_gram


def pca_deprecated(queryImage, featureModel, k):

    if( featureModel == "CM" ):
        #fetch the feature representaion CM
        X = color_moment_alternateV.combineColorMoments(queryImage)

        #standardize
        X_std = StandardScaler().fit_transform(X.reshape(-1,1))

    elif( featureModel == "HOG" ):
        X = skimage.feature.hog(queryImage, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')

        #standardize
        X_std = StandardScaler().fit_transform(X.reshape(-1,1))

    elif featureModel == "ELBP" :
        X = histogramFeatureDescriptorELBP(queryImage)
        X_std = StandardScaler().fit_transform(X.reshape(-1,1))




        


    assert( len(X) > 0 )

    #compute the covariance matrix
    meanFeatureVector = np.mean(X_std )
    covarianceMatrix = ((X_std.T - meanFeatureVector).T.dot((X_std.T - meanFeatureVector)))/(len(X_std) - 1)

    #eigen decomposition
    eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)

    #pre-process for sorting
    #eigenPairs : <eigenvalue, eigenvector> pair
    eigenPairs = []

    # print("vals", eigenValues)
    # print("pairs", eigenPairs)

    for i in range(len(eigenValues)):
        eigenPairs.append([np.abs(eigenValues[i]),eigenVectors[:,i]])
    
    #sort
    eigenPairs.sort(key=lambda x:x[0], reverse = False)

    #store top K latent sematics 
    topKLatentSemantics = []

    assert( k < len(X))

    for index in range(k):
        topKLatentSemantics.append(eigenPairs[index][0])


    '''YET TO IMPLEMENT THE SUBJECT-WEIGHT' PAIRS'''
    return topKLatentSemantics

def pca_task1(typeX, featureModel, k, eigen_id):
    '''typeX : {cc, con, detail, emboss, jitter, neg, noise01, noise02, original, poster, rot, smooth, stipple}'''

    typeImage = numpy.zeros((64,64))

    if( featureModel == "CM" ):
        list_files = os.listdir('./all_images')

        for fileName in list_files:
            if typeX in fileName:
                tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                typeImage = typeImage + tempImage

        #fetch the feature representaion CM
        X = color_moment_alternateV.combineColorMoments(typeImage)

        #standardize
        X_std = StandardScaler().fit_transform(X.reshape(-1,1))

    elif( featureModel == "HOG" ):

        list_files = os.listdir('./all_images')

        for fileName in list_files:
            if typeX in fileName:
                tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                typeImage = typeImage + tempImage

        X = skimage.feature.hog(typeImage, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')

        #standardize
        X_std = StandardScaler().fit_transform(X.reshape(-1,1))

    elif featureModel == "ELBP" :
        list_files = os.listdir('./all_images')

        for fileName in list_files:
            if typeX in fileName:
                tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                typeImage = typeImage + tempImage

        X = histogramFeatureDescriptorELBP(typeImage)
        X_std = StandardScaler().fit_transform(X.reshape(-1,1))




        


    assert( len(X) > 0 )

    #compute the covariance matrix
    meanFeatureVector = np.mean(X_std)
    covarianceMatrix = ((X_std.T - meanFeatureVector).T.dot((X_std.T - meanFeatureVector)))/(len(X_std) - 1)

    #eigen decomposition
    eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)

    #pre-process for sorting
    #eigenPairs : <eigenvalue, eigenvector> pair
    eigenPairs = []

    # print("vals", eigenValues)
    # print("pairs", eigenPairs)


    for i in range(len(eigenValues)):
        eigenPairs.append([np.abs(eigenValues[i]),eigenVectors[:,i]])

    #sort
    eigenPairs.sort(key=lambda x:x[0], reverse = True)
    
    print("eigenpairs", eigenPairs)
    image_to_db.eigen_to_db(eigenPairs, eigen_id)

    #store top K latent sematics 
    topKLatentSemantics = []

    # assert( k < len(X))

    for index in range(k):
        topKLatentSemantics.append(eigenPairs[index][0])


    subjects = []

    subjectImage = numpy.zeros((64,64))
    if ( featureModel == "CM" ):
        list_files = os.listdir('./all_images')

        for subjectID in range(40):
            subjectY = "-" + str(subjectID + 1) + "-"
            for fileName in list_files:
                if subjectY in fileName:
                    tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                    subjectImage = subjectImage + tempImage

            S = color_moment_alternateV.combineColorMoments(subjectImage)
            S_std = StandardScaler().fit_transform(S.reshape(-1,1))

            subjects.append(S_std)

    elif ( featureModel == "HOG" ):
        list_files = os.listdir('./all_images')

        for subjectID in range(40):
            subjectY = "-" + str(subjectID + 1) + "-"
            for fileName in list_files:
                if subjectY in fileName:
                    tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                    subjectImage = subjectImage + tempImage

            S = skimage.feature.hog(subjectImage, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
            S_std = StandardScaler().fit_transform(S.reshape(-1,1))
        
            subjects.append(S_std)

    elif ( featureModel == "ELBP" ):
        list_files = os.listdir('./all_images')

        for subjectID in range(40):
            subjectY = "-" + str(subjectID + 1) + "-"
            for fileName in list_files:
                if subjectY in fileName:
                    tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                    subjectImage = subjectImage + tempImage

            S = histogramFeatureDescriptorELBP(subjectImage)
            S_std = StandardScaler().fit_transform(S.reshape(-1,1))
        
            subjects.append(S_std)

    
    subjectWeightPairs = {}

    latentSemantics_subjectWeightpairs = []

    for latentSemantic in range(k):
    #     print("Latent Semantic", latentSemantic + 1 )
        for si in range(len(subjects)):
            subjectT = subjects[si].reshape(1,-1)[0]
    #         print("Subject ID:",si+1, " weight:", subjectT.dot(eigenPairs[latentSemantic][1]))
            subjectWeightPairs[si + 1] = subjectT.dot(eigenPairs[latentSemantic][1]).real
        subjectWeightPairs = dict(sorted(subjectWeightPairs.items(), key=lambda item: item[1], reverse=True))
        latentSemantics_subjectWeightpairs.append([latentSemantic + 1, subjectWeightPairs])
        subjectWeightPairs = {}
        


    '''Returns a (latentsemantic ,[subject ID, weight]) structure'''
    # return topKLatentSemantics
    return latentSemantics_subjectWeightpairs


def pca_task2(Y, featureModel, k, eigen_id):
    '''Y : 1 - 40'''

    subjectImage = numpy.zeros((64,64))

    if ( featureModel == "CM" ):
        list_files = os.listdir('./all_images')

        subjectY = "-" + str(Y) + "-"
        for fileName in list_files:
            if subjectY in fileName:
                tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                subjectImage = subjectImage + tempImage

        S = color_moment_alternateV.combineColorMoments(subjectImage)
        S_std = StandardScaler().fit_transform(S.reshape(-1,1))

    elif( featureModel == "HOG" ):

        list_files = os.listdir('./all_images')

        subjectY = "-" + str(Y) + "-"
        for fileName in list_files:
            if subjectY in fileName:
                tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                subjectImage = subjectImage + tempImage

        S = skimage.feature.hog(subjectImage, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
        S_std = StandardScaler().fit_transform(S.reshape(-1,1))
        

    elif featureModel == "ELBP" :
        list_files = os.listdir('./all_images')

        subjectY = "-" + str(Y) + "-"
        for fileName in list_files:
            if subjectY in fileName:
                tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                subjectImage = subjectImage + tempImage


        S = histogramFeatureDescriptorELBP(subjectImage)
        S_std = StandardScaler().fit_transform(S.reshape(-1,1))




        


    assert( len(S) > 0 )

    #compute the covariance matrix
    meanFeatureVector = np.mean(S_std)
    covarianceMatrix = ((S_std.T - meanFeatureVector).T.dot((S_std.T - meanFeatureVector)))/(len(S_std) - 1)

    #eigen decomposition
    eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)

    #pre-process for sorting
    #eigenPairs : <eigenvalue, eigenvector> pair
    eigenPairs = []

    # print("vals", eigenValues)
    # print("pairs", eigenPairs)

    for i in range(len(eigenValues)):
        eigenPairs.append([np.abs(eigenValues[i]),eigenVectors[:,i]])

    #sort
    eigenPairs.sort(key=lambda x:x[0], reverse = True)

    print("eigenpairs", eigenPairs)
    image_to_db.eigen_to_db(eigenPairs, eigen_id)

    #store top K latent sematics 
    topKLatentSemantics = []

    # assert( k < len(S))

    for index in range(k):
        topKLatentSemantics.append(eigenPairs[index][0])


    types = ['cc', 'con', 'emboss', 'jitter', 'neg', 'noise01', 'noise02', 'original', 'poster', 'rot', 'smooth', 'stipple']
    typesFD = []

    typeImage = numpy.zeros((64,64))
    if ( featureModel == "CM" ):
        list_files = os.listdir('./all_images')

        for typeID in range(12):
            for fileName in list_files:
                if types[typeID] in fileName:
                    tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                    typeImage = typeImage + tempImage

            Xx = color_moment_alternateV.combineColorMoments(typeImage)
            Xx_std = StandardScaler().fit_transform(Xx.reshape(-1,1))

            typesFD.append(Xx_std)

    elif ( featureModel == "HOG" ):
        list_files = os.listdir('./all_images')

        for typeID in range(12):
            for fileName in list_files:
                if types[typeID] in fileName:
                    tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                    typeImage = typeImage + tempImage

            Xx = skimage.feature.hog(typeImage, orientations = 9 , pixels_per_cell = (8,8) , cells_per_block=(3, 3), block_norm = 'L2-Hys')
            Xx_std = StandardScaler().fit_transform(Xx.reshape(-1,1))

            typesFD.append(Xx_std)

    elif ( featureModel == "ELBP" ):
        list_files = os.listdir('./all_images')

        for typeID in range(12):
            for fileName in list_files:
                if types[typeID] in fileName:
                    tempImage = io.imread(os.path.abspath('.') + "/all_images/" + fileName)
                    typeImage = typeImage + tempImage

            Xx = histogramFeatureDescriptorELBP(typeImage)
            Xx_std = StandardScaler().fit_transform(Xx.reshape(-1,1))

            typesFD.append(Xx_std)

    
    typeWeightPairs = {}

    latentSemantics_typeWeightpairs = []

    for latentSemantic in range(k):
    #     print("Latent Semantic", latentSemantic + 1 )
        for ti in range(len(typesFD)):
            typesT = typesFD[ti].reshape(1,-1)[0]
    #         print("Subject ID:",si+1, " weight:", subjectT.dot(eigenPairs[latentSemantic][1]))
            typeWeightPairs[types[ti]] = typesT.dot(eigenPairs[latentSemantic][1]).real
            #print(type(typesT.dot(eigenPairs[latentSemantic][1])))
        typeWeightPairs = dict(sorted(typeWeightPairs.items(), key=lambda item: item[1], reverse=True))
        latentSemantics_typeWeightpairs.append([latentSemantic + 1, typeWeightPairs])
        typeWeightPairs = {}
        


    '''Returns a (latentsemantic ,[type, weight]) structure'''
    return latentSemantics_typeWeightpairs




def pca_on_square_matrix(similarityMatrix, k):
    '''
    This assumes that a similarityMatrix has been computed already based on a feature descriptor
    '''

    #eigen decomposition
    eigenValues, eigenVectors = np.linalg.eig(similarityMatrix)

    #pre-process for sorting
    #eigenPairs : <eigenvalue, eigenvector> pair
    eigenPairs = []

    for i in range(len(eigenValues)):
        eigenPairs.append([np.abs(eigenValues[i]),eigenVectors[:,i]])

    #sort
    eigenPairs.sort(key=lambda x:x[0], reverse = True)

    #store top K latent sematics 
    topKLatentSemantics = []

    for index in range(k):
        topKLatentSemantics.append(eigenPairs[index][0])


    types = ['jitter', 'cc', 'con', 'emboss', 'neg', 'noise01', 'noise02', 'original', 'poster', 'rot', 'smooth', 'stipple']

    latentSemantics_typeWeightpairs = []

    for latentSemantic in range(len(eigenPairs)) :
#       print("Latent Semantic: ", latentSemantic + 1 )
        typeWeightPairs = []
    
        for ti in range(len(types)):
            typeWeightPairs.append([types[ti],eigenPairs[latentSemantic][1][ti]])
        
#       print(typeWeightPairs)
    
        typeWeightPairs.sort(key=lambda x:x[1], reverse = True)
    
        latentSemantics_typeWeightpairs.append([latentSemantic + 1, typeWeightPairs])
    


    return latentSemantics_typeWeightpairs




    




