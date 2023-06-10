import numpy as np
from skimage.io import imread
import phase1
import glob
import os
from PIL import Image
import math
import pickle
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from scipy.spatial.distance import cosine
from os.path import exists as file_exists
import matplotlib.pyplot as plt

# Original size of feature vectors
# CM - 192
# ELBP - 160
# HOG - 1764


# Function to perform PCA and returns top k latent features
# Assumes averages have already been done over Z
def pca(data, k):
    # k must be less than or equal to the original number of features
    if k > data.shape[1]:
        return -1
    # Compute covariance matrix
    cov = np.cov(data, rowvar=False)


    # Compute eigenvalues and eigenvectors (can use eigh since the covariance matrix is symmetric)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov)

    # For testing purposes only to compare eig vs eigh
    # test_vals, test_vecs = np.linalg.eig(cov)

    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    # test_pairs = [(np.abs(test_vals[i].real), test_vecs[:, i].real) for i in range(len(test_vals))]

    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    # test_pairs.sort(key=lambda k: k[0], reverse=True)

    U = [eigen_pairs[i][1][:, np.newaxis] for i in range(k)]
    # Add eigenvectors together to get m x k matrix
    U = np.hstack(U)

    # Extract top k eigenvalues and calculate square root
    S = [eigen_pairs[i][0] for i in range(k)]

    # print('Eigenvalues=', S)

    # Convert to diagonal matrix
    S = np.diag(S)

    return U, S

def svd(data, k):
    #k must be less than or equal to the original number of features
    if k > data.shape[1]:
        return -1

    # Compute D-D^T and D^T-D
    ddt = np.dot(data, data.T)
    dtd = np.dot(data.T, data)

    left_eigen_vals, left_eigen_vecs = np.linalg.eigh(ddt)
    right_eigen_vals, right_eigen_vecs = np.linalg.eigh(dtd)

    left_eigen_pairs = [(np.abs(left_eigen_vals[i]), left_eigen_vecs[:, i]) for i in range(len(left_eigen_vals))]
    left_eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    right_eigen_pairs = [(np.abs(right_eigen_vals[i]), right_eigen_vecs[:, i]) for i in range(len(right_eigen_vals))]
    right_eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    # Create matrix U from left eigenvectors
    U = [left_eigen_pairs[i][1][:, np.newaxis] for i in range(k)]
    # Add eigenvectors together to get n x k matrix
    U = np.hstack(U)

    # Extract top k left eigenvalues and calculate square root
    S = [np.sqrt(right_eigen_pairs[i][0]) for i in range(k)] #[(val1, vec1), (val2, vec2), (val3, vec3)]
    # Convert to diagonal matrix
    S = np.diag(S)

    # Create matrix V from right eigenvectors
    V = [right_eigen_pairs[i][1][:, np.newaxis] for i in range(k)]
    # Add eigenvectors together to get m x k matrix
    V = np.hstack(V)

    return U, S, V

# This method is called in task 1 when the user selects a value for X
# It returns a list of corresponding file names
def get_type_files(file_type):
    files = []
    for file_name in glob.iglob('all/image-' + file_type + '-*-*.png'):
        files.append(file_name)
    return files

# This method is called in task 2 when the user selects a value for Y
# It returns a list of corresponding file names
def get_subject_files(subject):
    files = []
    for file_name in glob.iglob('all/image-*-' + subject + '-*.png'):
        files.append(file_name)
    return files


# Used for tasks 3 and 4 to get all files
def get_all_files():
    files = []
    for file_name in glob.iglob('all/*.png'):
        files.append(file_name)
    return files



'''
TODO: figure out issue with nan results for elbp for noise02 images
'''
def get_type_type_mat(types_arr, files, folder_features, feature_model):
    #this groups the folder_features array by type
    type_features = []
    for type in range(len(types_arr)):
        type_feature = []
        for image in range(len(types_arr[type])):
            i = 0
            while i < len(files):
                if types_arr[type][image] == files[i]:
                    #if (type == 6):
                    #    print(folder_features[i])
                    type_feature.append(folder_features[i])
                    break
                else:
                    i += 1
        type_features.append(type_feature)
    #this groups each of the same indices across different feature vectors of the same type
    type_index_arr = []
    for type in range(len(type_features)):
        index_arr = []
        for index in range(len(type_features[type][0])):
            i = []
            i.append(type_features[type][0][index])
            index_arr.append(i)
        image = 1
        while image < len(type_features[type]):
            for index in range(len(type_features[type][image])):
                index_arr[index].append(type_features[type][image][index])
            image += 1
        type_index_arr.append(index_arr)
    #print(type_index_arr[6])
    #this finds the average value at each feature vector index for all feature vectors of the same type
    average_type_arr = []
    for type in range(len(type_index_arr)):
        average_feature = []
        for index in range(len(type_index_arr[type])):
            avg = np.average(type_index_arr[type][index])
            average_feature.append(avg)
        average_type_arr.append(average_feature)
    # print(range(len(average_type_arr)))
    # print(range(len(average_type_arr[0])))
    # compute similarity scores and create type-type matrix
    type_type_mat = []
    for type1 in range(len(average_type_arr)):
        type_sim_arr = []
        for type2 in range(len(average_type_arr)):
            if feature_model == "cm" or feature_model == "cm8x8":
                distance = phase1.euclideanDistance(np.asarray(average_type_arr[type1]), np.asarray(average_type_arr[type2]))
                if distance == 0:
                    sim = 1
                else:
                    sim = 1 / distance
            else:
                distance = phase1.earthMoversDistance(average_type_arr[type1], average_type_arr[type2])
                if distance == 0:
                    sim = 1
                else:
                    sim = 1 - distance
            type_sim_arr.append(sim)
        type_type_mat.append(type_sim_arr)
    # print(type_type_mat)
    # print(range(len(type_type_mat)))
    # print(range(len(type_type_mat[0])))
    return average_type_arr, type_type_mat

def get_subject_subject_mat(subjects_arr, files, folder_features, feature_model):
    # this groups the folder_features array by subject
    subject_features = []
    for subject in range(len(subjects_arr)):
        subject_feature = []
        for image in range(len(subjects_arr[subject])):
            i = 0
            while i < len(files):
                if subjects_arr[subject][image] == files[i]:
                    subject_feature.append(folder_features[i])
                    break
                else:
                    i += 1
        subject_features.append(subject_feature)
    # this groups each of the same indices across different feature vectors of the same type
    subject_index_arr = []
    #print(range(len(subject_features)))
    for subject in range(len(subject_features)):
        index_arr = []
        for index in range(len(subject_features[subject][0])):
            i = []
            i.append(subject_features[subject][0][index])
            index_arr.append(i)
        image = 1
        while image < len(subject_features[subject]):
            for index in range(len(subject_features[subject][image])):
                index_arr[index].append(subject_features[subject][image][index])
            image += 1
        subject_index_arr.append(index_arr)
    # this finds the average value at each feature vector index for all feature vectors of the same type
    average_subject_arr = []
    for subject in range(len(subject_index_arr)):
        average_feature = []
        for index in range(len(subject_index_arr[subject])):
            avg = np.average(subject_index_arr[subject][index])
            average_feature.append(avg)
        average_subject_arr.append(average_feature)
    # print(range(len(average_type_arr)))
    # print(range(len(average_type_arr[0])))
    # compute similarity scores and create type-type matrix
    subject_subject_mat = []
    for subject1 in range(len(average_subject_arr)):
        subject_sim_arr = []
        for subject2 in range(len(average_subject_arr)):
            if feature_model == "cm" or feature_model == "cm8x8":
                distance = phase1.euclideanDistance(np.asarray(average_subject_arr[subject1]),
                                                    np.asarray(average_subject_arr[subject2]))
                if distance == 0:
                    sim = 1
                else:
                    sim = 1 / distance
            else:
                distance = phase1.earthMoversDistance(average_subject_arr[subject1], average_subject_arr[subject2])
                if distance == 0:
                    sim = 1
                else:
                    sim = 1 - distance
            subject_sim_arr.append(sim)
        subject_subject_mat.append(subject_sim_arr)
    # print(subject_subject_mat)
    # print(range(len(subject_subject_mat)))
    # print(range(len(subject_subject_mat[0])))
    return average_subject_arr, subject_subject_mat


def choose_type():
    type = input('Enter Type (X):\n')
    k = int(input('Enter k:\n'))
    dim_red = input('Enter Dimensionality Reduction Technique:\n' +
                    '1 - PCA\n2 - SVD\n')

    if dim_red == '1':
        dim_red = "pca"
    elif dim_red == "2":
        dim_red = "svd"

    return type, k, dim_red


def choose_subject():
    subject = input('Enter Subject (Y):\n')
    k = int(input('Enter k:\n'))
    dim_red = input('Enter Dimensionality Reduction Technique:\n' +
                    '1 - PCA\n2 - SVD\n')

    if dim_red == '1':
        dim_red = "pca"
    elif dim_red == "2":
        dim_red = "svd"

    return subject, k, dim_red

# Function to extract all features from the files for task 1 and task 2
def get_features(feature_model, files):
    if feature_model == "cm" or feature_model == "cm8x8":
        extract_feature = phase1.color_moment
    elif feature_model == "elbp":
        extract_feature = phase1.elbp
    elif feature_model == "hog":
        extract_feature = phase1.my_hog
    else:
        print("Incorrect choice of model")
        exit()

    # Extract features
    folder_features = [extract_feature(imread(file)) for file in files]
    return folder_features

def task_5():
    query_filename = input('Enter path of query image:\n')
    latent_semantics_filename = input('Enter path of latent semantics file:\n')
    n = int(input('Enter n:\n'))
    # CASE 1: Input is file from task 1,2
    if latent_semantics_filename.split('-')[0] == '1' or latent_semantics_filename.split('-')[0] == '2':
        if latent_semantics_filename.split('-')[3] == 'pca':
            # Read in matrix U and scaler
            U_filename = latent_semantics_filename.split('-')
            U_filename[len(U_filename) - 1] = 'U'
            U_filename = '-'.join(U_filename)
            U = np.loadtxt(U_filename)

            scaler_filename = latent_semantics_filename.split('-')[:2]
            scaler_filename.append(latent_semantics_filename.split('-')[4])
            scaler_filename.append('scaler.joblib')
            scaler_filename = '-'.join(scaler_filename)
            scaler = load(scaler_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[4]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)


            # Standardize
            data_matrix = scaler.transform(data_matrix)

            # Apply U to matrix
            transformed_data_matrix = np.dot(data_matrix, U)

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            # Standardize
            query_feature = scaler.transform([query_feature])[0]

            # Apply U to query image
            query_feature = np.dot(query_feature, U)

            # Can only find 10 similar images for each type-subject
            num_pairs = int((n-1)/10)+1
            similarities = []
            for i in range(len(data_ordered_rows)):
                current_similarity = 1 - cosine(transformed_data_matrix[i], query_feature)
                similarities.append((data_ordered_rows[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            # Retrieve top num_pairs
            most_similar = []
            for i in range(num_pairs):
                most_similar.append(similarities[i][0])
            # print('MOST SIMILAR', most_similar)

            similar_files = []
            for sim in most_similar:
                for file_name in glob.iglob('all/image-' + sim.split('-')[0] + '-' + sim.split('-')[1] +'-*.png'):
                    similar_files.append(file_name)

            # Get features for similar files
            similar_features = get_features(feature_model, similar_files)

            # Convert to numpy array
            similar_features = np.asarray(similar_features)

            # Standardize
            similar_features = scaler.transform(similar_features)

            # Apply U to similar images
            similar_features = np.dot(similar_features, U)

            similarities = []
            for i in range(len(similar_files)):
                current_similarity = 1 - cosine(similar_features[i], query_feature)
                similarities.append((similar_files[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            return similarities[:n]
        elif latent_semantics_filename.split('-')[3] == 'svd':
            # Read in matrices V, S
            V_filename = latent_semantics_filename.split('-')
            V_filename[len(V_filename) - 1] = 'V'
            V_filename = '-'.join(V_filename)
            V = np.loadtxt(V_filename)

            S_filename = latent_semantics_filename.split('-')
            S_filename[len(S_filename) - 1] = 'S'
            S_filename = '-'.join(S_filename)
            S = np.loadtxt(S_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[4]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # No standardize

            # Apply V,S to matrix
            transformed_data_matrix = np.dot(np.dot(data_matrix, V), np.linalg.inv(S))
            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            # Apply V,S to query image
            query_feature = np.dot(np.dot(query_feature, V), np.linalg.inv(S))

            # Can only find 10 similar images for each type-subject
            num_pairs = int((n - 1) / 10) + 1
            similarities = []
            for i in range(len(data_ordered_rows)):
                current_similarity = 1 - cosine(transformed_data_matrix[i], query_feature)
                similarities.append((data_ordered_rows[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            # Retrieve top num_pairs
            most_similar = []
            for i in range(num_pairs):
                most_similar.append(similarities[i][0])
            # print('MOST SIMILAR', most_similar)

            similar_files = []
            for sim in most_similar:
                for file_name in glob.iglob('all/image-' + sim.split('-')[0] + '-' + sim.split('-')[1] + '-*.png'):
                    similar_files.append(file_name)

            # Get features for similar files
            similar_features = get_features(feature_model, similar_files)

            # Convert to numpy array
            similar_features = np.asarray(similar_features)

            # Apply V,S to similar images
            similar_features = np.dot(np.dot(similar_features, V), np.linalg.inv(S))

            similarities = []
            for i in range(len(similar_files)):
                current_similarity = 1 - cosine(similar_features[i], query_feature)
                similarities.append((similar_files[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            return similarities[:n]
    # CASE 2: Input is file from task 3
    elif latent_semantics_filename.split('-')[0] == '3':
        if latent_semantics_filename.split('-')[2] == 'pca':
            # Read in matrix U
            U_filename = latent_semantics_filename.split('-')
            U_filename[len(U_filename) - 1] = 'U'
            U_filename = '-'.join(U_filename)
            U = np.loadtxt(U_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[3]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # Read in type_feature file
            average_type_arr_filename = latent_semantics_filename.split('-')
            average_type_arr_filename [len(average_type_arr_filename ) - 1] = 'type_feature'
            average_type_arr_filename = '-'.join(average_type_arr_filename)

            average_type_arr = np.loadtxt(average_type_arr_filename)

            # Calculate distance to type for each row in type-subject (data_matrix)
            data_type_similarity = []
            for row in data_matrix:
                current_similarity_array = []
                for type_row in average_type_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                data_type_similarity.append(current_similarity_array)

            # Apply U to matrix
            data_type_similarity = np.asarray(data_type_similarity)
            data_type_similarity = np.dot(data_type_similarity, U)

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            # Get similarity of query_feature to types
            current_similarity_array = []
            for type_row in average_type_arr:
                if feature_model == "cm" or feature_model == "cm8x8":
                    distance = phase1.euclideanDistance(query_feature,
                                                        type_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 / distance
                else:
                    distance = phase1.earthMoversDistance(query_feature, type_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 - distance
                current_similarity_array.append(sim)

            query_feature = np.asarray(current_similarity_array)

            # Apply U to query image
            query_feature = np.dot(query_feature, U)

            # return data_type_similarity, query_feature, data_ordered_rows

            # Can only find 10 similar images for each type-subject
            num_pairs = int((n-1)/10)+1
            similarities = []
            for i in range(len(data_ordered_rows)):
                current_similarity = 1 - cosine(data_type_similarity[i], query_feature)
                similarities.append((data_ordered_rows[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            # Retrieve top num_pairs
            most_similar = []
            for i in range(num_pairs):
                most_similar.append(similarities[i][0])
            # print('MOST SIMILAR', most_similar)

            # Find which n images in best averages are most similar
            similar_files = []
            for sim in most_similar:
                for file_name in glob.iglob('all/image-' + sim.split('-')[0] + '-' + sim.split('-')[1] +'-*.png'):
                    similar_files.append(file_name)

            # Get features for similar files
            similar_features = get_features(feature_model, similar_files)

            # Convert to numpy array
            similar_features = np.asarray(similar_features)

            # Calculate distance to type for each row in most similar type-subject
            pair_similarities = []
            for row in similar_features:
                current_similarity_array = []
                for type_row in average_type_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                pair_similarities.append(current_similarity_array)

            # Apply U to similar images
            pair_similarities = np.asarray(pair_similarities)
            similar_features = np.dot(pair_similarities, U)

            similarities = []
            for i in range(len(similar_files)):
                current_similarity = 1 - cosine(similar_features[i], query_feature)
                similarities.append((similar_files[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            return similarities[:n]

        elif latent_semantics_filename.split('-')[2] == 'svd':
            # Read in matrix V,S
            V_filename = latent_semantics_filename.split('-')
            V_filename[len(V_filename) - 1] = 'V'
            V_filename = '-'.join(V_filename)
            V = np.loadtxt(V_filename)

            S_filename = latent_semantics_filename.split('-')
            S_filename[len(S_filename) - 1] = 'S'
            S_filename = '-'.join(S_filename)
            S = np.loadtxt(S_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[3]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # Read in type_feature file
            average_type_arr_filename = latent_semantics_filename.split('-')
            average_type_arr_filename [len(average_type_arr_filename ) - 1] = 'type_feature'
            average_type_arr_filename = '-'.join(average_type_arr_filename)

            average_type_arr = np.loadtxt(average_type_arr_filename)

            # Calculate distance to type for each row in type-subject (data_matrix)
            data_type_similarity = []
            for row in data_matrix:
                current_similarity_array = []
                for type_row in average_type_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                data_type_similarity.append(current_similarity_array)

            # Apply V,S to matrix
            data_type_similarity = np.asarray(data_type_similarity)
            data_type_similarity = np.dot(np.dot(data_type_similarity, V), np.linalg.inv(S))

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            # Get similarity of query_feature to types
            current_similarity_array = []
            for type_row in average_type_arr:
                if feature_model == "cm" or feature_model == "cm8x8":
                    distance = phase1.euclideanDistance(query_feature,
                                                        type_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 / distance
                else:
                    distance = phase1.earthMoversDistance(query_feature, type_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 - distance
                current_similarity_array.append(sim)

            query_feature = np.asarray(current_similarity_array)

            # Apply V,S to query image
            query_feature = np.dot(np.dot(query_feature, V), np.linalg.inv(S))


            # Can only find 10 similar images for each type-subject
            num_pairs = int((n-1)/10)+1
            similarities = []
            for i in range(len(data_ordered_rows)):
                current_similarity = 1 - cosine(data_type_similarity[i], query_feature)
                similarities.append((data_ordered_rows[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            # Retrieve top num_pairs
            most_similar = []
            for i in range(num_pairs):
                most_similar.append(similarities[i][0])
            # print('MOST SIMILAR', most_similar)

            # Find which n images in best averages are most similar
            similar_files = []
            for sim in most_similar:
                for file_name in glob.iglob('all/image-' + sim.split('-')[0] + '-' + sim.split('-')[1] +'-*.png'):
                    similar_files.append(file_name)

            # Get features for similar files
            similar_features = get_features(feature_model, similar_files)

            # Convert to numpy array
            similar_features = np.asarray(similar_features)

            # Calculate distance to type for each row in most similar type-subject
            pair_similarities = []
            for row in similar_features:
                current_similarity_array = []
                for type_row in average_type_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                pair_similarities.append(current_similarity_array)

            # Apply V,S to similar images
            pair_similarities = np.asarray(pair_similarities)
            similar_features = np.dot(np.dot(pair_similarities, V), np.linalg.inv(S))


            similarities = []
            for i in range(len(similar_files)):
                current_similarity = 1 - cosine(similar_features[i], query_feature)
                similarities.append((similar_files[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            return similarities[:n]
    # CASE 3: Input is file from task 4
    elif latent_semantics_filename.split('-')[0] == '4':
        if latent_semantics_filename.split('-')[2] == 'pca':
            # Read in matrix U
            U_filename = latent_semantics_filename.split('-')
            U_filename[len(U_filename) - 1] = 'U'
            U_filename = '-'.join(U_filename)
            U = np.loadtxt(U_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[3]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # Read in subject_feature file
            average_subject_arr_filename = latent_semantics_filename.split('-')
            average_subject_arr_filename[len(average_subject_arr_filename) - 1] = 'subject_feature'
            average_subject_arr_filename = '-'.join(average_subject_arr_filename)

            average_subject_arr = np.loadtxt(average_subject_arr_filename)

            # Calculate distance to subject for each row in type-subject (data_matrix)
            data_subject_similarity = []
            for row in data_matrix:
                current_similarity_array = []
                for subject_row in average_subject_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                data_subject_similarity.append(current_similarity_array)

            # Apply U to matrix
            data_subject_similarity = np.asarray(data_subject_similarity)
            data_subject_similarity = np.dot(data_subject_similarity, U)

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            # Get similarity of query_feature to types
            current_similarity_array = []
            for subject_row in average_subject_arr:
                if feature_model == "cm" or feature_model == "cm8x8":
                    distance = phase1.euclideanDistance(query_feature,
                                                        subject_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 / distance
                else:
                    distance = phase1.earthMoversDistance(query_feature, subject_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 - distance
                current_similarity_array.append(sim)

            query_feature = np.asarray(current_similarity_array)

            # Apply U to query image
            query_feature = np.dot(query_feature, U)


            # Can only find 10 similar images for each type-subject
            num_pairs = int((n - 1) / 10) + 1
            similarities = []
            for i in range(len(data_ordered_rows)):
                current_similarity = 1 - cosine(data_subject_similarity[i], query_feature)
                similarities.append((data_ordered_rows[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            # Retrieve top num_pairs
            most_similar = []
            for i in range(num_pairs):
                most_similar.append(similarities[i][0])
            # print('MOST SIMILAR', most_similar)

            # Find which n images in best averages are most similar
            similar_files = []
            for sim in most_similar:
                for file_name in glob.iglob('all/image-' + sim.split('-')[0] + '-' + sim.split('-')[1] + '-*.png'):
                    similar_files.append(file_name)

            # Get features for similar files
            similar_features = get_features(feature_model, similar_files)

            # Convert to numpy array
            similar_features = np.asarray(similar_features)

            # Calculate distance to subject for each row in most similar type-subject
            pair_similarities = []
            for row in similar_features:
                current_similarity_array = []
                for subject_row in average_subject_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                pair_similarities.append(current_similarity_array)

            # Apply U to similar images
            pair_similarities = np.asarray(pair_similarities)
            similar_features = np.dot(pair_similarities, U)

            similarities = []
            for i in range(len(similar_files)):
                current_similarity = 1 - cosine(similar_features[i], query_feature)
                similarities.append((similar_files[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            return similarities[:n]

        elif latent_semantics_filename.split('-')[2] == 'svd':
            # Read in matrix V,S
            V_filename = latent_semantics_filename.split('-')
            V_filename[len(V_filename) - 1] = 'V'
            V_filename = '-'.join(V_filename)
            V = np.loadtxt(V_filename)

            S_filename = latent_semantics_filename.split('-')
            S_filename[len(S_filename) - 1] = 'S'
            S_filename = '-'.join(S_filename)
            S = np.loadtxt(S_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[3]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # Read in type_feature file
            average_subject_arr_filename = latent_semantics_filename.split('-')
            average_subject_arr_filename[len(average_subject_arr_filename) - 1] = 'subject_feature'
            average_subject_arr_filename = '-'.join(average_subject_arr_filename)

            average_subject_arr = np.loadtxt(average_subject_arr_filename)

            # Calculate distance to subject for each row in type-subject (data_matrix)
            data_subject_similarity = []
            for row in data_matrix:
                current_similarity_array = []
                for subject_row in average_subject_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                data_subject_similarity.append(current_similarity_array)

            # Apply V,S to matrix
            data_subject_similarity = np.asarray(data_subject_similarity)
            data_subject_similarity = np.dot(np.dot(data_subject_similarity, V), np.linalg.inv(S))

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            # Get similarity of query_feature to types
            current_similarity_array = []
            for subject_row in average_subject_arr:
                if feature_model == "cm" or feature_model == "cm8x8":
                    distance = phase1.euclideanDistance(query_feature,
                                                        subject_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 / distance
                else:
                    distance = phase1.earthMoversDistance(query_feature, subject_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 - distance
                current_similarity_array.append(sim)

            query_feature = np.asarray(current_similarity_array)

            # Apply V,S to query image
            query_feature = np.dot(np.dot(query_feature, V), np.linalg.inv(S))

            # Can only find 10 similar images for each type-subject
            num_pairs = int((n - 1) / 10) + 1
            similarities = []
            for i in range(len(data_ordered_rows)):
                current_similarity = 1 - cosine(data_subject_similarity[i], query_feature)
                similarities.append((data_ordered_rows[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            # Retrieve top num_pairs
            most_similar = []
            for i in range(num_pairs):
                most_similar.append(similarities[i][0])
            # print('MOST SIMILAR', most_similar)

            # Find which n images in best averages are most similar
            similar_files = []
            for sim in most_similar:
                for file_name in glob.iglob('all/image-' + sim.split('-')[0] + '-' + sim.split('-')[1] + '-*.png'):
                    similar_files.append(file_name)

            # Get features for similar files
            similar_features = get_features(feature_model, similar_files)

            # Convert to numpy array
            similar_features = np.asarray(similar_features)

            # Calculate distance to subject for each row in most similar type-subject
            pair_similarities = []
            for row in similar_features:
                current_similarity_array = []
                for subject_row in average_subject_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                pair_similarities.append(current_similarity_array)

            # Apply V,S to similar images
            pair_similarities = np.asarray(pair_similarities)
            similar_features = np.dot(np.dot(pair_similarities, V), np.linalg.inv(S))

            similarities = []
            for i in range(len(similar_files)):
                current_similarity = 1 - cosine(similar_features[i], query_feature)
                similarities.append((similar_files[i], current_similarity))

            # Sort based on similarities
            similarities.sort(key=lambda k: k[1], reverse=True)

            return similarities[:n]


def task_6_7():
    query_filename = input('Enter path of query image:\n')
    latent_semantics_filename = input('Enter path of latent semantics file:\n')
    # CASE 1: Input is file from task 1,2
    # task, subject, str(k), dim_red, feature_model, "V"
    # task, subject, feature_model, 'scaler'))
    if latent_semantics_filename.split('-')[0] == '1' or latent_semantics_filename.split('-')[0] == '2':
        if latent_semantics_filename.split('-')[3] == 'pca':
            # Read in matrix U and scaler
            U_filename = latent_semantics_filename.split('-')
            U_filename[len(U_filename) - 1] = 'U'
            U_filename = '-'.join(U_filename)
            U = np.loadtxt(U_filename)

            scaler_filename = latent_semantics_filename.split('-')[:2]
            scaler_filename.append(latent_semantics_filename.split('-')[4])
            scaler_filename.append('scaler.joblib')
            scaler_filename = '-'.join(scaler_filename)
            scaler = load(scaler_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[4]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)


            # Standardize
            data_matrix = scaler.transform(data_matrix)

            # Apply U to matrix
            transformed_data_matrix = np.dot(data_matrix, U)

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            # Standardize
            query_feature = scaler.transform([query_feature])[0]

            # Apply U to query image
            query_feature = np.dot(query_feature, U)

            return transformed_data_matrix, query_feature, data_ordered_rows

        elif latent_semantics_filename.split('-')[3] == 'svd':
            # Read in matrices V, S
            V_filename = latent_semantics_filename.split('-')
            V_filename[len(V_filename) - 1] = 'V'
            V_filename = '-'.join(V_filename)
            V = np.loadtxt(V_filename)

            S_filename = latent_semantics_filename.split('-')
            S_filename[len(S_filename) - 1] = 'S'
            S_filename = '-'.join(S_filename)
            S = np.loadtxt(S_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[4]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # No standardize

            # Apply V,S to matrix
            transformed_data_matrix = np.dot(np.dot(data_matrix, V), np.linalg.inv(S))
            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            # Apply V,S to query image
            query_feature = np.dot(np.dot(query_feature, V), np.linalg.inv(S))

            return transformed_data_matrix, query_feature, data_ordered_rows
    # CASE 2: Input is file from task 3
    elif latent_semantics_filename.split('-')[0] == '3':
        if latent_semantics_filename.split('-')[2] == 'pca':
            # Read in matrix U
            U_filename = latent_semantics_filename.split('-')
            U_filename[len(U_filename) - 1] = 'U'
            U_filename = '-'.join(U_filename)
            U = np.loadtxt(U_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[3]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # Read in type_feature file
            average_type_arr_filename = latent_semantics_filename.split('-')
            average_type_arr_filename [len(average_type_arr_filename ) - 1] = 'type_feature'
            average_type_arr_filename = '-'.join(average_type_arr_filename)

            average_type_arr = np.loadtxt(average_type_arr_filename)

            # Calculate distance to type for each row in type-subject (data_matrix)
            data_type_similarity = []
            for row in data_matrix:
                current_similarity_array = []
                for type_row in average_type_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                data_type_similarity.append(current_similarity_array)

            # Apply U to matrix
            data_type_similarity = np.asarray(data_type_similarity)
            data_type_similarity = np.dot(data_type_similarity, U)

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            current_similarity_array = []
            for type_row in average_type_arr:
                if feature_model == "cm" or feature_model == "cm8x8":
                    distance = phase1.euclideanDistance(query_feature,
                                                        type_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 / distance
                else:
                    distance = phase1.earthMoversDistance(query_feature, type_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 - distance
                current_similarity_array.append(sim)

            query_feature = np.asarray(current_similarity_array)

            # Apply U to query image
            query_feature = np.dot(query_feature, U)

            return data_type_similarity, query_feature, data_ordered_rows

        elif latent_semantics_filename.split('-')[2] == 'svd':
            # Read in matrix V,S
            V_filename = latent_semantics_filename.split('-')
            V_filename[len(V_filename) - 1] = 'V'
            V_filename = '-'.join(V_filename)
            V = np.loadtxt(V_filename)

            S_filename = latent_semantics_filename.split('-')
            S_filename[len(S_filename) - 1] = 'S'
            S_filename = '-'.join(S_filename)
            S = np.loadtxt(S_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[3]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # Read in type_feature file
            average_type_arr_filename = latent_semantics_filename.split('-')
            average_type_arr_filename [len(average_type_arr_filename ) - 1] = 'type_feature'
            average_type_arr_filename = '-'.join(average_type_arr_filename)

            average_type_arr = np.loadtxt(average_type_arr_filename)

            # Calculate distance to type for each row in type-subject (data_matrix)
            data_type_similarity = []
            for row in data_matrix:
                current_similarity_array = []
                for type_row in average_type_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, type_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                data_type_similarity.append(current_similarity_array)

            # Apply V,S to matrix
            data_type_similarity = np.asarray(data_type_similarity)
            data_type_similarity = np.dot(np.dot(data_type_similarity, V), np.linalg.inv(S))

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            current_similarity_array = []
            for type_row in average_type_arr:
                if feature_model == "cm" or feature_model == "cm8x8":
                    distance = phase1.euclideanDistance(query_feature,
                                                        type_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 / distance
                else:
                    distance = phase1.earthMoversDistance(query_feature, type_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 - distance
                current_similarity_array.append(sim)

            query_feature = np.asarray(current_similarity_array)

            # Apply V,S to query image
            query_feature = np.dot(np.dot(query_feature, V), np.linalg.inv(S))

            return data_type_similarity, query_feature, data_ordered_rows

    # CASE 3: Input is file from task 4
    elif latent_semantics_filename.split('-')[0] == '4':
        if latent_semantics_filename.split('-')[2] == 'pca':
            # Read in matrix U
            U_filename = latent_semantics_filename.split('-')
            U_filename[len(U_filename) - 1] = 'U'
            U_filename = '-'.join(U_filename)
            U = np.loadtxt(U_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[3]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # Read in type_feature file
            average_subject_arr_filename = latent_semantics_filename.split('-')
            average_subject_arr_filename[len(average_subject_arr_filename) - 1] = 'subject_feature'
            average_subject_arr_filename  = '-'.join(average_subject_arr_filename )

            average_subject_arr = np.loadtxt(average_subject_arr_filename)

            # Calculate distance to subject for each row in type-subject (data_matrix)
            data_subject_similarity = []
            for row in data_matrix:
                current_similarity_array = []
                for subject_row in average_subject_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                data_subject_similarity.append(current_similarity_array)

            # Apply U to matrix
            data_subject_similarity = np.asarray(data_subject_similarity)
            data_subject_similarity  = np.dot(data_subject_similarity , U)

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            current_similarity_array = []
            for subject_row in average_subject_arr:
                if feature_model == "cm" or feature_model == "cm8x8":
                    distance = phase1.euclideanDistance(query_feature,
                                                        subject_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 / distance
                else:
                    distance = phase1.earthMoversDistance(query_feature, subject_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 - distance
                current_similarity_array.append(sim)

            query_feature = np.asarray(current_similarity_array)

            # Apply U to query image
            query_feature = np.dot(query_feature, U)

            return data_subject_similarity, query_feature, data_ordered_rows

        elif latent_semantics_filename.split('-')[2] == 'svd':
            # Read in matrix V,S
            V_filename = latent_semantics_filename.split('-')
            V_filename[len(V_filename) - 1] = 'V'
            V_filename = '-'.join(V_filename)
            V = np.loadtxt(V_filename)

            S_filename = latent_semantics_filename.split('-')
            S_filename[len(S_filename) - 1] = 'S'
            S_filename = '-'.join(S_filename)
            S = np.loadtxt(S_filename)

            # Read in type-subjectXfeature matrix
            feature_model = latent_semantics_filename.split('-')[3]
            if feature_model == 'cm':
                data_matrix = np.loadtxt('data-cm')
                with open('cm-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'elbp':
                data_matrix = np.loadtxt('data-elbp')
                with open('elbp-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)
            elif feature_model == 'hog':
                data_matrix = np.loadtxt('data-hog')
                with open('hog-rows.pickle', 'rb') as f:
                    data_ordered_rows = pickle.load(f)

            # Read in type_feature file
            average_subject_arr_filename = latent_semantics_filename.split('-')
            average_subject_arr_filename[len(average_subject_arr_filename) - 1] = 'subject_feature'
            average_subject_arr_filename = '-'.join(average_subject_arr_filename)

            average_subject_arr = np.loadtxt(average_subject_arr_filename)

            # Calculate distance to subject for each row in type-subject (data_matrix)
            data_subject_similarity = []
            for row in data_matrix:
                current_similarity_array = []
                for subject_row in average_subject_arr:
                    if feature_model == "cm" or feature_model == "cm8x8":
                        distance = phase1.euclideanDistance(row,
                                                            subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 / distance
                    else:
                        distance = phase1.earthMoversDistance(row, subject_row)
                        if distance == 0:
                            sim = 1
                        else:
                            sim = 1 - distance
                    current_similarity_array.append(sim)
                data_subject_similarity.append(current_similarity_array)

            # Apply V,S to matrix
            data_subject_similarity = np.asarray(data_subject_similarity)
            data_subject_similarity = np.dot(np.dot(data_subject_similarity, V), np.linalg.inv(S))

            # Get feature of query image
            query_feature = get_features(feature_model, [query_filename])[0]

            current_similarity_array = []
            for subject_row in average_subject_arr:
                if feature_model == "cm" or feature_model == "cm8x8":
                    distance = phase1.euclideanDistance(query_feature,
                                                        subject_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 / distance
                else:
                    distance = phase1.earthMoversDistance(query_feature, subject_row)
                    if distance == 0:
                        sim = 1
                    else:
                        sim = 1 - distance
                current_similarity_array.append(sim)

            query_feature = np.asarray(current_similarity_array)

            # Apply V,S to query image
            query_feature = np.dot(np.dot(query_feature, V), np.linalg.inv(S))

            return data_subject_similarity, query_feature, data_ordered_rows

def setup_data_matrices():
    if not file_exists('data-cm'):
        print('Setting up and storing cm data matrices to be used in later tasks')
        # Construct type-subjectXfeature matrix
        files = get_all_files()
        feature_model = 'cm'
        folder_features = get_features(feature_model, files)

        data_dict = {}

        for i in range(len(files)):
            current_row = files[i].split('-')[1] + '-' + files[i].split('-')[2]
            if current_row not in data_dict.keys():
                data_dict[current_row] = []
            data_dict[current_row].append(folder_features[i])

        data_matrix = []
        data_ordered_rows = []
        for key in data_dict.keys():
            data_ordered_rows.append(key)
            data_matrix.append(np.average((data_dict[key]), axis=0))

        data_matrix = np.asarray(data_matrix)
        np.savetxt('data-cm', data_matrix)

        with open('cm-rows.pickle', 'wb') as f:
            pickle.dump(data_ordered_rows, f)

    if not file_exists('data-elbp'):
        print('Setting up and storing elbp data matrices to be used in later tasks')
        # Construct type-subjectXfeature matrix
        files = get_all_files()
        feature_model = 'elbp'
        folder_features = get_features(feature_model, files)

        data_dict = {}

        for i in range(len(files)):
            current_row = files[i].split('-')[1] + '-' + files[i].split('-')[2]
            if current_row not in data_dict.keys():
                data_dict[current_row] = []
            data_dict[current_row].append(folder_features[i])

        data_matrix = []
        data_ordered_rows = []
        for key in data_dict.keys():
            data_ordered_rows.append(key)
            data_matrix.append(np.average((data_dict[key]), axis=0))

        data_matrix = np.asarray(data_matrix)
        np.savetxt('data-elbp', data_matrix)

        with open('elbp-rows.pickle', 'wb') as f:
            pickle.dump(data_ordered_rows, f)

    if not file_exists('data-hog'):
        print('Setting up and storing hog data matrices to be used in later tasks')
        # Construct type-subjectXfeature matrix
        files = get_all_files()
        feature_model = 'hog'
        folder_features = get_features(feature_model, files)

        data_dict = {}

        for i in range(len(files)):
            current_row = files[i].split('-')[1] + '-' + files[i].split('-')[2]
            if current_row not in data_dict.keys():
                data_dict[current_row] = []
            data_dict[current_row].append(folder_features[i])

        data_matrix = []
        data_ordered_rows = []
        for key in data_dict.keys():
            data_ordered_rows.append(key)
            data_matrix.append(np.average((data_dict[key]), axis=0))

        data_matrix = np.asarray(data_matrix)
        np.savetxt('data-hog', data_matrix)

        with open('hog-rows.pickle', 'wb') as f:
            pickle.dump(data_ordered_rows, f)
def main():
    # CODE TO AVERAGE: np.average((cms[0], cms[1], cms[2]), axis=0)
    quit = False
    while quit == False:
        setup_data_matrices()
        print('Enter number of task to perform:')
        print('1. Task 1')
        print('2. Task 2')
        print('3. Task 3')
        print('4. Task 4')
        print('5. Task 5')
        print('6. Task 6')
        print('7. Task 7')
        print('8. Task 8')
        print('9. View image types')
        print('10. Quit')
        task = input()
        if task == '9':
            # files = get_type_files('cc')
            # f = files[0]
            # print(files[0].split('-'))
            #

            print('original', (imread('all/image-original-1-1.png')))
            print('emboss', (imread('all/image-emboss-1-1.png')))
            print('neg', (imread('all/image-neg-1-1.png')))
            print('poster',(imread('all/image-poster-1-1.png')))
            print('jitter',(imread('all/image-jitter-1-1.png')))
            print('cc',(imread('all/image-cc-1-1.png')))
            print('noise01', (imread('all/image-noise01-1-1.png')))
            print('noise02', (imread('all/image-noise02-1-1.png')))
            print('stipple',(imread('all/image-stipple-1-1.png')))
            print('smooth',(imread('all/image-smooth-1-1.png')))
            print('con',(imread('all/image-con-1-1.png')))
            print('rot',(imread('all/image-rot-1-1.png')))

        if task == '1':
            feature_model = input('Enter feature model:\n').lower()
            type, k, dim_red = choose_type()
            files = get_type_files(type)
            # For testing correct files found
            # for f in files:
            #     print(f)
            #     break
            folder_features = get_features(feature_model, files)

            # Dictionary to store arrays based on subject
            subject_dict = {}  # 1 : [[feature vector1],[feature vector2]]
            # Take average across subjects
            for i in range(len(files)):
                current_subject = files[i].split('-')[2]
                if current_subject not in subject_dict.keys():
                    subject_dict[current_subject] = []
                # Add feature vector to array for this subject
                subject_dict[current_subject].append(folder_features[i])

            # print(subject_dict.keys())
            subject_matrix = []
            subject_ordered_rows = []
            for key in subject_dict.keys():
                subject_ordered_rows.append("S" + key)
                subject_matrix.append(np.average((subject_dict[key]), axis=0))

            # subject_matrix = [np.average((subject_dict[key]), axis=0) for key in subject_dict.keys()]
            # print(len(subject_matrix))
            # print(len(subject_matrix[0]))
            # print(len(subject_matrix[1]))

            subject_matrix = np.asarray(subject_matrix)

            if dim_red == '1' or dim_red.lower() == 'pca':
                # SCALING/PREPROCESSING
                scaler = StandardScaler()
                subject_matrix = scaler.fit_transform(subject_matrix)

                # Save scaler
                scaler_filename = "-".join((task, type, feature_model, 'scaler.joblib'))
                dump(scaler, scaler_filename)

                U, S = pca(subject_matrix, k)

                # Transform subject-feature matrix into subject-latent features by applying U
                transformed_subject_matrix = np.dot(subject_matrix, U)

                subject_weight_pairs = []
                for column in range(transformed_subject_matrix.shape[1]):
                    current_column_pairs = [(subject_ordered_rows[i], transformed_subject_matrix[i, column]) for i in
                                            range(len(subject_ordered_rows))]
                    # Order pairs in descending order
                    current_column_pairs.sort(key=lambda k: abs(k[1]), reverse=True)

                    subject_weight_pairs.append(current_column_pairs)

                output_string = ''
                for arr in subject_weight_pairs:
                    output_string += str(arr) + '\n'
                print(output_string)

                # Output matrices to files
                subj_weight_filename = "-".join((task, type, str(k), dim_red, feature_model, "latent"))
                with open(subj_weight_filename, 'w') as f:
                    f.write(output_string)

                U_filename = "-".join((task, type, str(k), dim_red, feature_model, "U"))
                np.savetxt(U_filename, U)

                # # Compare to PCA
                # from sklearn.decomposition import PCA
                # my_pca = PCA(n_components=5)
                # x_pca = my_pca.fit_transform(subject_matrix)
                #
                # print('S', np.diagonal(S))
                # print('pca_explained_variance', my_pca.explained_variance_)
                #
                # print('U', U[:5])
                # print('pca_components', my_pca.components_.T[:5])
                #
                # print(transformed_subject_matrix.shape)
                # print(x_pca.shape)
                #
                # print(transformed_subject_matrix[1:5, :])
                # print(x_pca[1:5, :])

            elif dim_red == '2' or dim_red.lower() == 'svd':
                U, S, V = svd(subject_matrix, k)

                # Sign of U and data*V*S^-1 might be swapped b/c left eigenvectors and right eigenvectors are solved for seperately
                # To solve this, just construct U from S and V
                U = np.dot(np.dot(subject_matrix, V), np.linalg.inv(S))

                # print(U[:, 0])
                subject_weight_pairs = []
                for column in range(U.shape[1]):
                    current_column_pairs = [(subject_ordered_rows[i], U[i, column]) for i in
                                            range(len(subject_ordered_rows))]
                    # Order pairs in descending order
                    current_column_pairs.sort(key=lambda k: abs(k[1]), reverse=True)

                    subject_weight_pairs.append(current_column_pairs)

                # print(subject_weight_pairs[0])
                # print(len(subject_weight_pairs[0]))

                output_string = ''
                for arr in subject_weight_pairs:
                    output_string += str(arr) + '\n'
                print(output_string)

                # Output matrices to files
                subj_weight_filename = "-".join((task, type, str(k), dim_red, feature_model, "U"))
                with open(subj_weight_filename, 'w') as f:
                    f.write(output_string)

                S_filename = "-".join((task, type, str(k), dim_red, feature_model, "S"))
                np.savetxt(S_filename, S)

                V_filename = "-".join((task, type, str(k), dim_red, feature_model, "V"))
                np.savetxt(V_filename, V)

                # A check for making sure the matrices are the same
                print('--U--\n',U[:3])
                print('--RECONSTRUCTED--\n', np.dot(np.dot(subject_matrix, V), np.linalg.inv(S))[:3])

        if task == '2':
            feature_model = input('Enter feature model:\n').lower()
            subject, k, dim_red = choose_subject()
            files = get_subject_files(subject)
            # For testing correct files found
            # for f in files:
            #     print(f)
            folder_features = get_features(feature_model, files)

            # Dictionary to store arrays based on type
            type_dict = {}
            # Take average across types
            for i in range(len(files)):
                current_type = files[i].split('-')[1]
                if current_type not in type_dict.keys():
                    type_dict[current_type] = []
                # Add feature vector to array for this subject
                type_dict[current_type].append(folder_features[i])

            type_matrix = []
            type_ordered_rows = []
            for key in type_dict.keys():
                type_ordered_rows.append(key)
                type_matrix.append(np.average((type_dict[key]), axis=0))

            type_matrix = np.asarray(type_matrix)

            if dim_red == '1' or dim_red.lower() == 'pca':
                # SCALING/PREPROCESSING
                scaler = StandardScaler()
                type_matrix = scaler.fit_transform(type_matrix)

                # Save scaler
                scaler_filename = "-".join((task, subject, feature_model, 'scaler.joblib'))
                dump(scaler, scaler_filename)

                U, S = pca(type_matrix, k)

                # Transform subject-feature matrix into subject-latent features by applying U
                transformed_type_matrix = np.dot(type_matrix, U)
                # print(transformed_subject_matrix)

                type_weight_pairs = []
                for column in range(transformed_type_matrix.shape[1]):
                    current_column_pairs = [(type_ordered_rows[i], transformed_type_matrix[i, column]) for i in
                                            range(len(type_ordered_rows))]
                    # Order pairs in descending order
                    current_column_pairs.sort(key=lambda k: abs(k[1]), reverse=True)

                    type_weight_pairs.append(current_column_pairs)

                output_string = ''
                for arr in type_weight_pairs:
                    output_string += str(arr) + '\n'
                print(output_string)

                # Output matrices to files
                type_weight_filename = "-".join((task, subject, str(k), dim_red, feature_model, "latent"))
                with open(type_weight_filename, 'w') as f:
                    f.write(output_string)

                U_filename = "-".join((task, subject, str(k), dim_red, feature_model, "U"))
                np.savetxt(U_filename, U)

                # # Compare to PCA
                # from sklearn.decomposition import PCA
                # my_pca = PCA(n_components=5)
                # x_pca = my_pca.fit_transform(type_matrix)
                #
                # # print('S', S)
                # print('pca_explained_variance', my_pca.explained_variance_)
                #
                # print('U', U[:5])
                # print('pca_components', my_pca.components_.T[:5])
                #
                # print(transformed_type_matrix.shape)
                # print(x_pca.shape)
                #
                # print(transformed_type_matrix[1:5, :])
                # print(x_pca[1:5, :])
            elif dim_red == '2' or dim_red.lower() == 'svd':
                U, S, V = svd(type_matrix, k)

                # Sign of U and data*V*S^-1 might be swapped b/c left eigenvectors and right eigenvectors are solved for seperately
                # To solve this, just construct U from S and V
                U = np.dot(np.dot(type_matrix, V), np.linalg.inv(S))

                # print(U[:, 0])
                type_weight_pairs = []
                for column in range(U.shape[1]):
                    current_column_pairs = [(type_ordered_rows[i], U[i, column]) for i in range(len(type_ordered_rows))]
                    # Order pairs in descending order
                    current_column_pairs.sort(key=lambda k: k[1], reverse=True)

                    type_weight_pairs.append(current_column_pairs)

                # print(subject_weight_pairs[0])
                # print(len(subject_weight_pairs[0]))

                output_string = ''
                for arr in type_weight_pairs:
                    output_string += str(arr) + '\n'
                print(output_string)

                # Output matrices to files
                subj_weight_filename = "-".join((task, subject, str(k), dim_red, feature_model, "U"))
                with open(subj_weight_filename, 'w') as f:
                    f.write(output_string)

                S_filename = "-".join((task, subject, str(k), dim_red, feature_model, "S"))
                np.savetxt(S_filename, S)

                V_filename = "-".join((task, subject, str(k), dim_red, feature_model, "V"))
                np.savetxt(V_filename, V)

                # print('--U--\n', U)
                # print('--RECONSTRUCTED--\n', np.dot(np.dot(type_matrix, V), np.linalg.inv(S)))

        if task == '3':
            feature_model = input('Enter feature model:\n').lower()
            k = int(input('Enter k:\n'))
            dim_red = input('Enter Dimensionality Reduction Technique:\n' +
                            '1 - PCA\n2 - SVD\n')

            if dim_red == '1':
                dim_red = "pca"
            elif dim_red == "2":
                dim_red = "svd"

            files = get_all_files()
            folder_features = get_features(feature_model, files)
            types = ['cc', 'con', 'emboss', 'jitter', 'neg', 'noise01', 'noise02', 'original', 'poster', 'rot', 'smooth', 'stipple']
            types_arr = []
            for i in range(len(types)):
                type = get_type_files(types[i])
                types_arr.append(type)

            average_type_arr, type_type_mat = get_type_type_mat(types_arr, files, folder_features, feature_model)
            # Save type_type
            np.savetxt("-".join((task, str(k), dim_red, feature_model, 'type_type')), type_type_mat)
            # Save average_type_arr
            np.savetxt("-".join((task, str(k), dim_red, feature_model, 'type_feature')), average_type_arr)

            if(dim_red == '1' or dim_red == 'pca'):
                U, S = pca(np.asarray(type_type_mat), k)

                # Reduce dimensionality of type-type similarity matrix
                dim_red_data = np.dot(np.asarray(type_type_mat), U)

                type_weight_pairs = []
                for column in range(dim_red_data.shape[1]):
                    current_column_pairs = [(types[i], dim_red_data[i, column]) for i in
                                            range(len(types))]
                    # Order pairs in descending order
                    current_column_pairs.sort(key=lambda k: abs(k[1]), reverse=True)

                    type_weight_pairs.append(current_column_pairs)

                output_string = ''
                for arr in type_weight_pairs:
                    output_string += str(arr) + '\n'
                print(output_string)

                # Output matrices to files
                type_weight_filename = "-".join((task, str(k), dim_red, feature_model, "latent"))
                with open(type_weight_filename, 'w') as f:
                    f.write(output_string)

                U_filename = "-".join((task, str(k), dim_red, feature_model, "U"))
                np.savetxt(U_filename, U)
            else:
                U, S, V = svd(np.asarray(type_type_mat), k)

                dim_red_data = np.dot(np.dot(np.asarray(type_type_mat), V), np.linalg.inv(S))

                type_weight_pairs = []
                for column in range(U.shape[1]):
                    current_column_pairs = [(types[i], dim_red_data[i, column]) for i in
                                            range(len(types))]
                    # Order pairs in descending order
                    current_column_pairs.sort(key=lambda k: abs(k[1]), reverse=True)

                    type_weight_pairs.append(current_column_pairs)

                output_string = ''
                for arr in type_weight_pairs:
                    output_string += str(arr) + '\n'
                print(output_string)

                # Output matrices to files
                type_weight_filename = "-".join((task, str(k), dim_red, feature_model, "U"))
                with open(type_weight_filename, 'w') as f:
                    f.write(output_string)

                S_filename = "-".join((task, str(k), dim_red, feature_model, "S"))
                np.savetxt(S_filename, S)

                V_filename = "-".join((task, str(k), dim_red, feature_model, "V"))
                np.savetxt(V_filename, V)
            # print(dim_red_data)

        if task == '4':
            feature_model = input('Enter feature model:\n').lower()
            k = int(input('Enter k:\n'))
            dim_red = input('Enter Dimensionality Reduction Technique:\n' +
                            '1 - PCA\n2 - SVD\n')

            if dim_red == '1':
                dim_red = "pca"
            elif dim_red == "2":
                dim_red = "svd"

            files = get_all_files()
            folder_features = get_features(feature_model, files)
            subjects = np.arange(1, 41, 1)
            subjects_arr = []
            for i in range(len(subjects)):
                subject = get_subject_files(str(subjects[i]))
                subjects_arr.append(subject)
            average_subject_arr, subject_subject_mat = get_subject_subject_mat(subjects_arr, files, folder_features, feature_model)

            # Save subject_subject
            np.savetxt("-".join((task, str(k), dim_red, feature_model, 'subject_subject')), subject_subject_mat)
            # Save average_subject_arr
            np.savetxt("-".join((task, str(k), dim_red, feature_model, 'subject_feature')), average_subject_arr)

            if (dim_red == '1' or dim_red == 'pca'):
                U, S = pca(np.asarray(subject_subject_mat), k)

                # Reduce dimensionality of subject-subject similarity matrix
                dim_red_data = np.dot(np.asarray(subject_subject_mat), U)

                subject_weight_pairs = []
                for column in range(dim_red_data.shape[1]):
                    current_column_pairs = [(subjects[i], dim_red_data[i, column]) for i in
                                            range(len(subjects))]
                    # Order pairs in descending order
                    current_column_pairs.sort(key=lambda k: abs(k[1]), reverse=True)

                    subject_weight_pairs.append(current_column_pairs)

                output_string = ''
                for arr in subject_weight_pairs:
                    output_string += str(arr) + '\n'
                print(output_string)

                # Output matrices to files
                subject_weight_filename = "-".join((task, str(k), dim_red, feature_model, "latent"))
                with open(subject_weight_filename, 'w') as f:
                    f.write(output_string)

                U_filename = "-".join((task, str(k), dim_red, feature_model, "U"))
                np.savetxt(U_filename, U)
            else:
                U, S, V = svd(np.asarray(subject_subject_mat), k)

                dim_red_data = np.dot(np.dot(np.asarray(subject_subject_mat), V), np.linalg.inv(S))

                subject_weight_pairs = []
                for column in range(U.shape[1]):
                    current_column_pairs = [(subjects[i], dim_red_data[i, column]) for i in
                                            range(len(subjects))]
                    # Order pairs in descending order
                    current_column_pairs.sort(key=lambda k: abs(k[1]), reverse=True)

                    subject_weight_pairs.append(current_column_pairs)

                output_string = ''
                for arr in subject_weight_pairs:
                    output_string += str(arr) + '\n'
                print(output_string)

                # Output matrices to files
                subject_weight_filename = "-".join((task, str(k), dim_red, feature_model, "U"))
                with open(subject_weight_filename, 'w') as f:
                    f.write(output_string)

                S_filename = "-".join((task, str(k), dim_red, feature_model, "S"))
                np.savetxt(S_filename, S)

                V_filename = "-".join((task, str(k), dim_red, feature_model, "V"))
                np.savetxt(V_filename, V)
            # print(dim_red_data)
        if task == '5':
            # IMPLEMENT TASK 5
            # similarities.append((similar_files[i], current_similarity))
            similarities = task_5()
            print(similarities)
            n = len(similarities)

            page_num = 1
            for i in range(0,n,4):
                fig = plt.figure(page_num)
                for j in range(i,i+4):
                    if j == n:
                        break
                    img_grid = fig.add_subplot(2, 2, j-i+1)
                    img_grid.imshow(imread(similarities[j][0]))
                    img_grid.set_ylabel('Score: ' + str(similarities[j][1]))
                    img_grid.set_title(str(j+1))
                    img_grid.set_xlabel(similarities[j][0])

                fig.suptitle('Query Results: ' + str(page_num))
                fig.set_size_inches(10, 10)
                fig.subplots_adjust(hspace=0.53)
                page_num += 1

            plt.show()
        if task == '6':
            transformed_data_matrix, query_feature, data_ordered_rows = task_6_7()

            # Find most similar type-subject to query image in k-dimensional space
            highest_similarity = 0
            for i in range(len(data_ordered_rows)):
                current_similarity = 1 - cosine(transformed_data_matrix[i], query_feature)
                if current_similarity > highest_similarity:
                    highest_similarity = current_similarity
                    most_similar = data_ordered_rows[i]
            print(highest_similarity)
            print(most_similar.split('-')[0])

        if task == '7':
            transformed_data_matrix, query_feature, data_ordered_rows = task_6_7()
            # Find most similar type-subject to query image in k-dimensional space
            highest_similarity = 0
            for i in range(len(data_ordered_rows)):
                current_similarity = 1 - cosine(transformed_data_matrix[i], query_feature)
                if current_similarity > highest_similarity:
                    highest_similarity = current_similarity
                    most_similar = data_ordered_rows[i]
            print(highest_similarity)
            print(most_similar.split('-')[1])

        if task == '8':
            subject_subject_filename = input('Enter path of subject-subject similarity matrix:\n')
            n = int(input('Value for n:\n'))
            m = int(input('Enter value for m:\n'))
            subject_subject_mat = np.loadtxt(subject_subject_filename)

            print('Creating similarity graph')

            new_graph_pairs = []
            for k in range(subject_subject_mat.shape[0]):
                # Find n most similar nodes
                row = subject_subject_mat[k]
                # print(row)
                most_similar = []
                for i in range(len(row)):
                    if i == k:
                        continue
                    if len(most_similar) < n:
                        most_similar.append((i, row[i]))
                    else:
                        most_similar.sort(key=lambda k: k[1], reverse=True)
                        if row[i] > most_similar[n-1][1]:
                            most_similar.pop(n-1)
                            most_similar.append((i, row[i]))
                new_graph_pairs.append(most_similar)

            # Construct new graph for first part of 8
            new_graph = []
            for row in new_graph_pairs:
                current_row = [0] * 40
                for pair in row:
                    current_row[pair[0]] = pair[1]
                new_graph.append(current_row)

            new_graph = np.asarray(new_graph)

            part1_filename = '8-' + str(n) + '-' + subject_subject_filename.split('-')[3]
            np.savetxt(part1_filename, new_graph)
            print('Done!')

            # Part 2: ASCOS++
            print('Calculating ASCOS++')
            # Randomly intialize graph
            ascos_graph = np.random.rand(40,40)


            c = 0.8
            for iterations in range(1500):
                previous_value = np.array(ascos_graph)
                for i in range(ascos_graph.shape[0]):
                    for j in range(ascos_graph.shape[1]):
                        if i == j:
                            ascos_graph[i][j] = 1
                        else:
                            # Sum all in-neighbors of i
                            in_neighbors = []
                            for k in range(new_graph.shape[0]):
                                if new_graph[k][i] != 0:
                                    in_neighbors.append(k)
                            # Calculate i*
                            w_istar = 0
                            for k in in_neighbors:
                                w_istar += new_graph[i][k]

                            neighbor_summation = 0
                            # Update s_ij
                            for k in in_neighbors:
                                if w_istar == 0:
                                    break
                                neighbor_summation += (new_graph[i][k]/w_istar) * (1-math.exp(-1*new_graph[i][k])) * previous_value[k][j]
                            # Multiply by c
                            neighbor_summation *= c
                    ascos_graph[i][j] = neighbor_summation
                # Check for convergence
                if np.array_equal(ascos_graph, previous_value):
                    print('Stopping early at ' + str(iterations) + ' iterations')
                    break


            # print(ascos_graph)
            # Save graph
            ascoss_graph_filename = '8-' + str(n) + '-' + str(m) + '-' + 'graph' + '-' + \
                                    subject_subject_filename.split('-')[3]
            np.savetxt(ascoss_graph_filename, ascos_graph)

            node_scores = ascos_graph.sum(axis=0) / 40

            # Find largest m
            largest_m = []
            for i in range(node_scores.shape[0]):
                if len(largest_m) < m:
                    largest_m.append((i, node_scores[i]))
                else:
                    largest_m.sort(key=lambda k: k[1], reverse=True)
                    if node_scores[i] > largest_m[m-1][1]:
                        largest_m.pop(m-1)
                        largest_m.append((i+1, node_scores[i]))

            # Sort one more time
            largest_m.sort(key=lambda k: k[1], reverse=True)

            most_important_subjects = []
            for pair in largest_m:
                most_important_subjects.append(pair[0])

            print('Most important subjects:\n')
            print(most_important_subjects)
            # Save most important subjects and the importance values to file
            part2_filename =  '8-' + str(n) + '-' + str(m) + '-' + subject_subject_filename.split('-')[3]
            with open(part2_filename, 'w') as f:
                f.write(str(largest_m))
        if task == '10':
            quit = True

if __name__ == '__main__':
    main()

#     for file_name in glob.iglob('all/image-*-1-*.png'):
#         print(file_name)
#     exit()
# #     with os.scandir('all') as it:
# #         for entry in it:
# #             if entry.name.startswith('image-cc'):
# #                 print(entry.name)
# #     exit()
#
#     # main()
#
#     data = imread('all/image-cc-1-1.png')
#
#     cm_1 = phase1.color_moment(data)
#
#     data = imread('all/image-cc-1-2.png')
#     cm_2 = phase1.color_moment(data)
#
#     data = imread('all/image-cc-1-3.png')
#     cm_3 = phase1.color_moment(data)
#
#     data = imread('all/image-cc-1-4.png')
#     cm_4 = phase1.color_moment(data)
#
#     data = imread('all/image-cc-1-5.png')
#     cm_5 = phase1.color_moment(data)
#
#     data = imread('all/image-cc-1-6.png')
#     cm_6 = phase1.color_moment(data)
#
#     data = imread('all/image-cc-1-7.png')
#     cm_7 = phase1.color_moment(data)
#
#     data = imread('all/image-cc-1-8.png')
#     cm_8 = phase1.color_moment(data)
#
#     data = imread('all/image-cc-1-9.png')
#     cm_9 = phase1.color_moment(data)
#
#     data = np.stack((cm_1, cm_2, cm_3, cm_4, cm_5, cm_6, cm_7, cm_8, cm_9), axis=0)
#
#     cov = np.cov(data, rowvar=False)
#
#     # Compute eigenvalues and eigenvectors (can use eigh since the covariance matrix is symmetric)
#     eigen_vals, eigen_vecs = np.linalg.eigh(cov)
#
#     # For testing purposes only to compare eig vs eigh
#     test_vals, test_vecs = np.linalg.eig(cov)
#
#     eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
#     test_pairs = [(np.abs(test_vals[i].real), test_vecs[:, i].real) for i in range(len(test_vals))]
#
#     eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#     test_pairs.sort(key=lambda k: k[0], reverse=True)
#
#     k = 5
#     U = [eigen_pairs[i][1][:, np.newaxis] for i in range(k)]
#     # Add eigenvectors together to get m x k matrix
#     T = np.hstack(U)
#
#     # Convert objects to k-dim space
#     # np.dot is same as np.matmul
#     reduced_data = np.matmul(data, T)
    
    
    
    
# eigh WAY faster than eig
# =============================================================================
# import time
# t= time.time()
# # Compute eigenvalues and eigenvectors (can use eigh since the covariance matrix is symmetric)
# eigen_vals, eigen_vecs = np.linalg.eigh(cov)
# print(time.time()-t)
# 
# # For testing purposes only to compare eig vs eigh
# t = time.time()
# test_vals, test_vecs = np.linalg.eig(cov)
# print(time.time()-t)
# =============================================================================

