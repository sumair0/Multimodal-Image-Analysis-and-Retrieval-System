from sklearn import datasets
import numpy as np
import scipy
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog
# from skimage.feature import hog
from skimage.io import imread
import glob
import os
from scipy.stats import wasserstein_distance
import ntpath
from scipy.spatial.distance import cosine
from scipy.special import rel_entr
import math
import pickle

# Function to calculate color moments of an image
def color_moment(image):
    feature = []
    # Start from the bottom right/end and work its way back to the top left/beginning of image and continually prepend
    for row in range(0, 57, 8):
        for col in range(0, 57, 8):
            values = image[row:row + 8, col:col + 8]
            # print(values)
            mean = np.mean(values)
            calculated_std = 0
            calculated_skewness = 0

            for num in values.flatten():
                calculated_std += (num - mean) ** 2
                calculated_skewness += (num - mean) ** 3
            calculated_std = (calculated_std / 64) ** (1 / 2)
            # Calculate the cube root of the absolute value of the variable to get the skewness. The first term just reapplies the sign (+/-) to the result
            if calculated_skewness != 0:
                calculated_skewness = int(calculated_skewness / abs(calculated_skewness)) * (
                            abs(calculated_skewness) / 64) ** (1 / 3)

            feature.append(mean)
            feature.append(calculated_std)
            feature.append(calculated_skewness)
            # print(feature)
    return np.asarray(feature)


def elbp(image):
    # Calculate uniform LBP
    lbpu = local_binary_pattern(image, 8, 1, 'uniform')
    # Calculate variance
    lbpv = local_binary_pattern(image, 8, 1, 'var')
    # Bins for 2-D histogram (see calc_var_bins.py for derivation of bins for variance)
    lbp_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    var_bins = [0.0, 0.00001, 1.3827439210981538, 7.296454629780101, 16.59799359628596, 31.324201114113635, 54.65285737325328,
                90.51102535185555, 146.2263469365662, 236.5093576081257, 364.2331429883261, 530.4661701920563, 781.5244790204242,
                1292.028400509851, 2288.524008988099, 6793.375323594772, 12570.768848111478]

    # var_bins = [0.0, 2.3420074916527938, 4.9597703118092795, 8.295748475028176, 12.555121737481386, 18.051872273956178,
    #             25.16856667983984, 34.547447553750544, 47.03868615696774, 64.07498734477423, 88.00294140949154,
    #             122.87955648325055, 176.1619694556398, 265.35077964955246, 443.3784170852141, 1057.9593746247647,
    #             9722.143494012096]

    hist, x_edges, y_edges = np.histogram2d(lbpu.flatten(), lbpv.flatten(), bins=(lbp_bins, var_bins))
    # Not entirely sure about the order of the coordinates, something to look into if there is time
    # Only order is really important for converting to feature vector
    # print(hist.flatten())
    # print(x_edges)
    # print(y_edges)
    return hist.flatten()


def my_hog(image):
    hist = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return hist


def get_folder_images():
    images = []
    ids = []
    folder_path = input('Enter relative path for folder:\n')

    for file_name in glob.glob(folder_path + '\*.png'):
        # Get 64X64 grayscale values for the current image in the folder
        images.append(imread(file_name))
        ids.append(ntpath.basename(file_name).split('.')[0])

    return images, ids, folder_path


# A method to return the sorted lists of image ids in the folder based on their resulting distances from the query image
# Params:
# image: The list of images in the folder
# image_ids: The list of image_ids in the folder
# distances: The list of distances from the query image to every other image in the folder
# similarity: A flag to indicate that distances is a list of similarity values and the resulting lists should therefore be sorted
def sort_distances(images, image_ids, distances, similarity=False):
    # Zip the lists together with distances as the first value used for sorting
    zipped_lists = zip(distances, image_ids, images)

    sorted_zipped = sorted(zipped_lists, reverse=similarity)

    sorted_distances, sorted_ids, images = [list(lst) for lst in zip(*sorted_zipped)]

    return images, sorted_ids, sorted_distances



# Function for plotting images
# images: is a list of the numpy arrays in the properly sorted order based on distance
# scores: a list of the properly sorted distances
# total_images: total number of images to be printed out
# image_ids: a list of the image ids sorted by distance
# rows: number of rows
# cols: number of cols
def plot_images(images, scores, image_ids, total_images, rows, cols):
    fig = plt.figure()
    for i in range(total_images):
        img_grid = fig.add_subplot(rows, cols, i+1)
        img_grid.imshow(images[i])
        img_grid.set_ylabel('Score: ' + str(scores[i]))
        img_grid.set_title(str(i+1))
        img_grid.set_xlabel(image_ids[i])

    fig.suptitle('Query Results')
    fig.set_size_inches(10, rows*5)
    fig.subplots_adjust(hspace=0.53)
    # fig.tight_layout(pad=1.0)
    return fig
#         plt.figure()
#         plt.suptitle('Query Image')
#         plt.imshow(query_img)
#         plt.savefig(query_results_filename + '-query.png')
#         plt.show()
def sort_distance_extra(images, image_ids, distances, elbp_dists, hog_dists):
    zipped_lists = zip(distances, image_ids, images, elbp_dists, hog_dists)

    sorted_zipped = sorted(zipped_lists)

    sorted_distances, sorted_ids, images, sorted_elbp, sorted_hog = [list(lst) for lst in zip(*sorted_zipped)]

    return images, sorted_ids, sorted_distances, sorted_elbp, sorted_hog


def euclideanDistance(data1, data2):
    distance = np.sqrt(np.sum(np.square(data1 - data2)))
    return distance

def earthMoversDistance(data1, data2):
    distance = wasserstein_distance(data1 / np.sum(data1), data2 / np.sum(data2))
    return distance

def main():
    olivetti = datasets.fetch_olivetti_faces()

    images = olivetti.images
    data = olivetti.data
    targets = olivetti.target

    np.set_printoptions(threshold=np.inf)
    #


    # Image ID as combination of subject ID and index
    # Could comprise of just index, but following in accordance with Professor's Piazza post
    im_id = []
    for ind, subj in enumerate(targets):
        im_id.append(str(subj) + '-' + str(ind % 10))

    task = input('Enter Task:\n')

    if task == '1':
        id = input('Enter Image ID:\n')

        try:
            image_index = im_id.index(id)
        except ValueError:
            print('Invalid Image ID entered')
            exit()

        model = input('Enter model:\n').lower()

        # Determine model to use
        if model == 'cm' or model == 'cm8x8':
            # print('CM8X8 selected')
            extract_feature = color_moment
        elif model == 'elbp':
            # print('ELBP selected')
            extract_feature = elbp
        elif model == 'hog':
            # print('HOG selected')
            extract_feature = my_hog
        else:
            print("Incorrect choice of model")
            exit()

        # print(np.round((images[image_index]*255)))
        # exit()
        feature = extract_feature(np.round(images[image_index] * 255))
        print(feature)
    elif task == '2':
        # Extract image data and image IDs from input folder
        folder_images, folder_ids, folder_path = get_folder_images()
        folder_features = []  # A list of 3-tuples (CM8x8, ELBP, HOG)

        # Calculate feature descriptors for all images
        for img in folder_images:
            current_cm = color_moment(img)
            current_elbp = elbp(img)
            current_hog = my_hog(img)

            # Append the current image's features
            folder_features.append((current_cm, current_elbp, current_hog))

        output_string = ''
        for id, feature_tuple in zip(folder_ids, folder_features):
            output_string += ('*' * 15)
            output_string += '\n' + str(id) + '\n'
            output_string += ('*' * 15)

            output_string += '\nCM8X8:\n'
            output_string += str(feature_tuple[0])
            output_string += '\n'

            output_string += 'ELBP\n'
            output_string += str(feature_tuple[1])
            output_string += '\n'

            output_string += 'HOG\n'
            output_string += str(feature_tuple[2])
            output_string += '\n\n\n'

        print(output_string)

        output_filename = folder_path + '-features.txt'
        with open(output_filename, 'wb') as f:
            f.write(output_string)

    elif task == '5':
        # Extract image data and image IDs from input folder
        folder_images, folder_ids, folder_path = get_folder_images()

        # Get feature model to use
        model = input('Enter feature model:\n')

        # Determine model to use
        if model == 'cm' or model == 'cm8x8':
            # print('CM8X8 selected')
            extract_feature = color_moment
        elif model == 'elbp':
            # print('ELBP selected')
            extract_feature = elbp
        elif model == 'hog':
            # print('HOG selected')
            extract_feature = my_hog
        else:
            print("Incorrect choice of model")
            exit()

        folder_features = np.asarray([extract_feature(img) for img in folder_images])
        output_filename = '1-' + folder_path + '-' + model
        np.savetxt(output_filename, folder_features)

        with open(output_filename+'.pkl', 'wb') as f:
            pickle.dump(folder_ids, f)
    elif task == '3':
        # Extract image data and image IDs from input folder
        folder_images, folder_ids, folder_path = get_folder_images()

        query_feature = 0
        folder_features = []
        folder_dists = []

        query_id = input('Enter Image ID\n')

        # Get index of input image
        try:
            query_index = folder_ids.index(query_id)
        except ValueError:
            print('Invalid Image ID entered')
            exit()

        # Store query image
        query_img = folder_images[query_index]

        # Remove query image from images
        folder_images.pop(query_index)

        # Remove image ID of query image
        folder_ids.pop(query_index)

        model = input('Enter model:\n').lower()

        # Determine model to use
        if model == 'cm' or model == 'cm8x8':
            # print('CM8X8 selected')
            extract_feature = color_moment
        elif model == 'elbp':
            # print('ELBP selected')
            extract_feature = elbp
        elif model == 'hog':
            # print('HOG selected')
            extract_feature = my_hog
        else:
            print("Incorrect choice of model")
            exit()

        k = int(input('Enter value for k:\n'))

        query_feature = extract_feature(query_img)
        folder_features = [extract_feature(img) for img in folder_images]

        if model == 'cm' or model == 'cm8x8':
            folder_dists = [np.linalg.norm((obj - query_feature),2) for obj in folder_features]


        elif model == 'elbp':
            folder_dists = [wasserstein_distance(query_feature/4096, obj/4096) for obj in folder_features]


        elif model == 'hog':
            folder_dists = [wasserstein_distance(query_feature/np.sum(query_feature), obj/np.sum(obj)) for obj in folder_features]


        folder_images, folder_ids, folder_dists = sort_distances(folder_images, folder_ids, folder_dists)
        for index in range(k):
            print(folder_ids[index], folder_dists[index])

        query_results_filename = folder_path + '-' + query_id + '-' + str(k) + '-' + model +'.png'

        fig = plot_images(folder_images, folder_dists, folder_ids, k, math.ceil(k/2), 2)
        fig.savefig(query_results_filename)

        # Print query image
        plt.figure()
        plt.suptitle('Query Image')
        plt.imshow(query_img)
        plt.savefig(query_results_filename + '-query.png')
        plt.show()

    elif task == '4':
        # Extract image data and image IDs from input folder
        folder_images, folder_ids, folder_path = get_folder_images()

        query_feature = 0
        folder_features_tuples = []
        folder_dists = []

        query_id = input('Enter Image ID\n')

        # Get index of input image
        try:
            query_index = folder_ids.index(query_id)
        except ValueError:
            print('Invalid Image ID entered')
            exit()

        k = int(input('Enter value for k:\n'))

        # Store query image
        query_img = folder_images[query_index]

        # Remove query image from images
        folder_images.pop(query_index)

        # Remove image ID of query image
        folder_ids.pop(query_index)

        query_feature = (elbp(query_img), my_hog(query_img))
        folder_features = [(elbp(img), my_hog(img)) for img in folder_images]

        elbp_dists = [wasserstein_distance(query_feature[0]/4096, obj[0]/4096) for obj in
                        folder_features]
        hog_dists = [wasserstein_distance(query_feature[1] / np.sum(query_feature[1]), obj[1] / np.sum(obj[1])) for obj in
                        folder_features]

        folder_dists = [(elbp_dist + hog_dist*10)/2 for elbp_dist, hog_dist in zip(elbp_dists, hog_dists)]

        folder_images, folder_ids, folder_dists, elbp_dists, hog_dists = sort_distance_extra(folder_images, folder_ids, folder_dists, elbp_dists, hog_dists)
        for index in range(k):
            print(folder_ids[index], folder_dists[index])
            print('Contributions')
            print('ELBP: ', elbp_dists[index])
            print('HOG: ', hog_dists[index])
            print()
        query_results_filename = folder_path + '-' + query_id + '-' + str(k) + '.png'

        fig = plot_images(folder_images, folder_dists, folder_ids, k, math.ceil(k/2), 2)
        fig.savefig(query_results_filename)

        # Print query image
        plt.figure()
        plt.suptitle('Query Image')
        plt.imshow(query_img)
        plt.savefig(query_results_filename + '-query.png')
        plt.show()

if __name__ == '__main__':
    main()
