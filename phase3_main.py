# Imports
from phase1 import *
from phase2 import *
from skimage.io import imread
from sklearn.preprocessing import StandardScaler
import numpy as np
from PPRClassifier import PPRClassifier
import decisionTree
from VAIndex import VAIndex
import lsh2

class_array = [''] * 12
class_array[0] = 'cc'
class_array[1] = 'con'
class_array[2] = 'emboss'
class_array[3] = 'jitter'
class_array[4] = 'neg'
class_array[5] = 'noise01'
class_array[6] = 'noise02'
class_array[7] = 'original'
class_array[8] = 'poster'
class_array[9] = 'rot'
class_array[10] = 'smooth'
class_array[11] = 'stipple'

# Function for computing false positive and miss rates
# Inputs:
def get_classifier_results(predictions, labels, label_type, classifier_type='dt'):
    if len(predictions) != len(labels):
        raise Exception('Size of prediction list and size of label list are not equal')

    # Each label will need to retain TP, FN, FP, TN
    label_dict = {}
    if label_type == 'X':
        label_dict['cc'] = [0,0,0,0]
        label_dict['con'] = [0,0,0,0]
        label_dict['emboss'] = [0,0,0,0]
        label_dict['jitter'] = [0,0,0,0]
        label_dict['neg'] = [0,0,0,0]
        label_dict['noise01'] = [0,0,0,0]
        label_dict['noise02'] = [0,0,0,0]
        label_dict['original'] = [0,0,0,0]
        label_dict['poster'] = [0,0,0,0]
        label_dict['rot'] = [0,0,0,0]
        label_dict['smooth'] = [0,0,0,0]
        label_dict['stipple'] = [0,0,0,0]
    elif label_type == 'Y':
        for i in range(1,41):
            label_dict[str(i)] = [0,0,0,0]
    elif label_type == 'Z':
        for i in range(1,11):
            label_dict[str(i)] = [0,0,0,0]
    # print(predictions)
    # Indices Order: TP, FN, FP, TN
    for pred_label, true_label in zip(predictions, labels):
        if classifier_type == 'ppr':
            pred_label = pred_label
        else:
            if label_type == 'X':
                pred_label = class_array[pred_label]
            else:
                pred_label = pred_label + 1
                pred_label = str(pred_label)


        if pred_label == true_label:
            label_dict[true_label][0] += 1
        else:
            # Add False Negative
            label_dict[true_label][1] += 1
            # Add False Positive
            label_dict[pred_label][2] += 1
        # Add True Negative
        for key in label_dict.keys():
            if key != pred_label and key != true_label:
                label_dict[key][3] += 1

    print('Label\tFalse Positive Rate\tMiss Rate')
    print('-'*50)

    # Compute and print false positive and miss rates
    sorted_keys = sorted(list(label_dict.keys()))
    #print(label_dict.keys())
    #print(type(label_dict.keys()))
    for key in sorted_keys:
        if label_dict[key][2] == 0:
            fp_rate = 0
        else:
            fp_rate = label_dict[key][2]/(label_dict[key][2]+label_dict[key][3])
        if label_dict[key][1] == 0:
            miss_rate = 0
        else:
            miss_rate = label_dict[key][1]/(label_dict[key][0]+label_dict[key][1])
        # print(key + '\t' + str(fp_rate) + '\t' + str(miss_rate))

        # we can use the % rates later but I think this is more helpful for now
        print(key + '\t' + str(label_dict[key][2]) + '/' + str((label_dict[key][2]+label_dict[key][3])) + 
        '\t' + str(label_dict[key][1]) + '/' + str(label_dict[key][0]+label_dict[key][1]))

# Function to get relevant data for tasks 1-3
def get_info():
    folder1 = input('Enter path for first folder:\n')
    feature_model = input('Enter feature model:\n').lower()
    k = int(input('Enter k:\n'))
    folder2 = input('Enter path for second folder:\n')

    files1 = []
    for file_name in glob.iglob(folder1 + '/*.png'):
        files1.append(file_name)

    files2 = []
    for file_name in glob.iglob(folder2 + '/*.png'):
        files2.append(file_name)

    print('Enter classifier model to be used: ')
    print('1. SVM')
    print('2. Decision Tree')
    print('3. PPR')
    classifier = input()

    return files1, feature_model, k, files2, classifier


def main():

    quit = False
    while quit == False:
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
        if task == '10':
            quit = True
        elif task == '1':
            files1, feature_model, k, files2, classifier = get_info()
            if feature_model == "cm" or feature_model == "cm8x8":
                extract_feature = phase1.color_moment
            elif feature_model == "elbp":
                extract_feature = phase1.elbp
            elif feature_model == "hog":
                extract_feature = phase1.my_hog

            # Get features for files in both folders
            folder_features_1 = np.asarray([extract_feature(imread(file)) for file in files1])
            folder_features_2 = np.asarray([extract_feature(imread(file)) for file in files2])

            # Get X labels for training and testing data
            training_labels = np.asarray([file.split('-')[1] for file in files1])
            testing_labels = np.asarray([file.split('-')[1] for file in files2])

            # Scale the data for PCA
            scaler = StandardScaler()
            training_data = scaler.fit_transform(folder_features_1)
            testing_data = scaler.transform(folder_features_2)

            # Apply PCA
            U, S = pca(training_data, k)

            # Transform training data matrix into latent features space by applying U
            training_data = np.dot(training_data, U)

            # Transform testing data matrix into latent features space by applying U
            testing_data = np.dot(testing_data, U)

            #print('space')
            #print(training_data)
            #print(len(training_data))
            #print(len(training_labels))
            if classifier == '1':
                print('svm')
            elif classifier == '2':
                #print('decision tree')
                #clf = decisionTree.DecisionTree(max_depth = 3)
                clf = decisionTree.DecisionTreeClassifier(max_depth=12,task1=True)
                clf.fit(training_data, training_labels)
                pred = clf.predict(testing_data)
                get_classifier_results(pred, testing_labels, 'X')
            elif classifier == '3':
                pprClassifier = PPRClassifier(0.85)
                pprClassifier.addTrainingData(training_data, list(training_labels))
                pprClassifier.addTestingData(testing_data)
                pprClassifier.setupGraph()
                prediction_labels = pprClassifier.predict()
                get_classifier_results(prediction_labels, testing_labels, 'X', 'ppr')
            else:
                return -1
        elif task == '2':
            files1, feature_model, k, files2, classifier = get_info()
            if feature_model == "cm" or feature_model == "cm8x8":
                extract_feature = phase1.color_moment
            elif feature_model == "elbp":
                extract_feature = phase1.elbp
            elif feature_model == "hog":
                extract_feature = phase1.my_hog

            # Get features for files in both folders
            folder_features_1 = np.asarray([extract_feature(imread(file)) for file in files1])
            folder_features_2 = np.asarray([extract_feature(imread(file)) for file in files2])

            # Get Y labels for training and testing data
            training_labels = np.asarray([file.split('-')[2] for file in files1])
            testing_labels = np.asarray([file.split('-')[2] for file in files2])

            # Scale the data for PCA
            scaler = StandardScaler()
            training_data = scaler.fit_transform(folder_features_1)
            testing_data = scaler.transform(folder_features_2)

            # Apply PCA
            U, S = pca(training_data, k)

            # Transform training data matrix into latent features space by applying U
            training_data = np.dot(training_data, U)

            # Transform testing data matrix into latent features space by applying U
            testing_data = np.dot(testing_data, U)

            if classifier == '1':
                print('svm')
            elif classifier == '2':
                print('decision tree')
                clf = decisionTree.DecisionTreeClassifier(max_depth=40)
                clf.fit(training_data, training_labels)
                pred = clf.predict(testing_data)
                get_classifier_results(pred, testing_labels, 'Y')                
            elif classifier == '3':
                pprClassifier = PPRClassifier(0.85)
                pprClassifier.addTrainingData(training_data, list(training_labels))
                pprClassifier.addTestingData(testing_data)
                pprClassifier.setupGraph()
                prediction_labels = pprClassifier.predict()
                get_classifier_results(prediction_labels, testing_labels, 'Y', 'ppr')
            else:
                return -1
        elif task == '3':
            files1, feature_model, k, files2, classifier = get_info()
            if feature_model == "cm" or feature_model == "cm8x8":
                extract_feature = phase1.color_moment
            elif feature_model == "elbp":
                extract_feature = phase1.elbp
            elif feature_model == "hog":
                extract_feature = phase1.my_hog

            # Get features for files in both folders
            folder_features_1 = np.asarray([extract_feature(imread(file)) for file in files1])
            folder_features_2 = np.asarray([extract_feature(imread(file)) for file in files2])

            # Get Z labels for training and testing data

            training_labels = np.asarray([file.split('-')[3].split('.')[0] for file in files1])
            testing_labels = np.asarray([file.split('-')[3].split('.')[0] for file in files2])

            # Scale the data for PCA
            scaler = StandardScaler()
            training_data = scaler.fit_transform(folder_features_1)
            testing_data = scaler.transform(folder_features_2)

            # Apply PCA
            U, S = pca(training_data, k)

            # Transform training data matrix into latent features space by applying U
            training_data = np.dot(training_data, U)

            # Transform testing data matrix into latent features space by applying U
            testing_data = np.dot(testing_data, U)

            if classifier == '1':
                print('svm')
            elif classifier == '2':
                print('decision tree')
                clf = decisionTree.DecisionTreeClassifier(max_depth=10)
                clf.fit(training_data, training_labels)
                pred = clf.predict(testing_data)
                get_classifier_results(pred, testing_labels, 'Z')   
            elif classifier == '3':
                pprClassifier = PPRClassifier(0.85)
                pprClassifier.addTrainingData(training_data, list(training_labels))
                pprClassifier.addTestingData(testing_data)
                pprClassifier.setupGraph()
                prediction_labels = pprClassifier.predict()
                get_classifier_results(prediction_labels, testing_labels, 'Z', 'ppr')
            else:
                return -1
        elif task == '4':
            folder = input("Enter the filepath of the folder to perform LSH on: ")
            files = []
            for file_name in glob.iglob(folder + '/*.png'):
                files.append(file_name)

            feature_model = input("Enter feature model: ")
            if feature_model == "cm" or feature_model == "cm8x8":
                extract_feature = phase1.color_moment
            elif feature_model == "elbp":
                extract_feature = phase1.elbp
            elif feature_model == "hog":
                extract_feature = phase1.my_hog
            folder_features = np.asarray([extract_feature(imread(file)) for file in files])

            k = int(input("Enter k value: "))
            layers = int(input("Enter number of Layers: "))
            image_path = input("Enter filepath of image to perform search on: ")
            image_feature = extract_feature(imread(image_path))
            t = input("Enter t value: ")
            lsh_result = lsh2.lsh(folder_features, k, layers, image_feature, t, files)
        elif task == '5':
            b = input('Enter b:\n')
            if b == 'q':
                model = vaIndex.get_model()
                # Get image and t
                image_path = input('Enter path of query image:\n')
                t = int(input('Enter value of t:\n'))

                feature_model = vaIndex.get_model()
                if feature_model == "cm" or feature_model == "cm8x8":
                    extract_feature = phase1.color_moment
                elif feature_model == "elbp":
                    extract_feature = phase1.elbp
                elif feature_model == "hog":
                    extract_feature = phase1.my_hog

                query_features = extract_feature(imread(image_path))
                num_candidates_checked, file_indices = vaIndex.query(query_features, t)
                print('Total number of actual vectors compared', num_candidates_checked)

                print('\nPredictions:')
                for index in file_indices:
                    print(file_names[index])
            else:
                vector_filepath = input('Enter vector filepath:\n')
                # Case 1: Input is from phase 1
                if vector_filepath.split('-')[0] == '1':
                    vectors = np.loadtxt(vector_filepath)
                    with open(vector_filepath + '.pkl', 'rb') as f:
                        file_names = pickle.load(f)
                    model = vector_filepath.split('-')[2]

                    # Create index
                    vaIndex = VAIndex(b, vectors, model)

                # Get image and t
                image_path = input('Enter path of query image:\n')
                t = int(input('Enter value of t:\n'))

                feature_model = vaIndex.get_model()
                if feature_model == "cm" or feature_model == "cm8x8":
                    extract_feature = phase1.color_moment
                elif feature_model == "elbp":
                    extract_feature = phase1.elbp
                elif feature_model == "hog":
                    extract_feature = phase1.my_hog

                query_features = extract_feature(imread(image_path))
                num_candidates_checked, file_indices = vaIndex.query(query_features, t)
                print('Total number of actual vectors compared', num_candidates_checked)

                print('\nPredictions:')
                for index in file_indices:
                    print(file_names[index])

if __name__ == '__main__':
    main()
