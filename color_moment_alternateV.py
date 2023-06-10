import numpy as numpy
import scipy as scipy
from scipy.stats import skew
import matplotlib.pyplot as plt
from skimage import io
import os
import skimage.feature


def computeMean( imagePortion ) :
    return (numpy.sum(imagePortion))/((len(imagePortion))*(len(imagePortion[0])))
    
def computeStandardDeviation( imagePortion ) :
    return numpy.std(imagePortion)

def computeSkewness( imagePortion ):
    return scipy.stats.skew(imagePortion)

# Divides the image into 8x8 blocks and computes color moments for each block 
def computeColorMoment(image):

    npimage = numpy.array(image)        
    meanMatrix = numpy.zeros((8 ,8))
    standardDeviationMatrix = numpy.zeros((8 ,8))
    skewnessMatrix = numpy.zeros((8 ,8))
    
    
    row = 0 
    column = 0
    step = 8
    size = 64
    
    while(row <= (size - 8) and column <= (size - 8)):         
        meanMatrix[int(row/8)][int(column/8)] = computeMean(npimage[row:row + step , column:column + step])
        standardDeviationMatrix[int(row/8)][int(column/8)] = computeStandardDeviation(npimage[row:row + step , column:column + step])
        skewnessMatrix[int(row/8)][int(column/8)] = computeSkewness(npimage[row:row + step , column:column + step].flatten())

        
        column += step
        
        if(column > (size- 8)):
            row += step
            column = 0
            
    return meanMatrix,standardDeviationMatrix,skewnessMatrix


#use this method as your calling function
def combineColorMoments(image):
    [meanMatrix, standardDeviationMatrix, skewnessMatrix] = computeColorMoment(image)
    
    combinedMean = numpy.sum(meanMatrix)
    combinedStandardDeviation = numpy.sum(standardDeviationMatrix)
    combinedSkewness = numpy.sum(skewnessMatrix)
    
    return numpy.array([combinedMean, combinedStandardDeviation, combinedSkewness])