from sklearn import datasets
from skimage.feature import local_binary_pattern
import numpy as np
from skimage.io import imread
import phase2

files = phase2.get_all_files()


variances = []

for file in files:
    img = imread(file)
    image_var = local_binary_pattern(img, 8, 1, 'var')
    variances.append(image_var)

print(len(variances))
variances = np.asarray(variances)
print(variances.shape)

# Convert NAN to 0.0
variances = np.nan_to_num(variances)


# Extract 0th-16th quantile of the data
quantiles = []
for i in range(17):
    quantiles.append(1/16*i)

var_bins = []
# Compute the bins for equally splitting the variance data
for q in quantiles:
    var_bins.append(np.quantile(variances, q))
print(var_bins)


# # Construct histogram
# hist, bin_edges = np.histogram(variances, bins=var_bins)
#
# print(hist)
# print((bin_edges))

print()
var_bins[1] = 1e-5
# Construct histogram
hist, bin_edges = np.histogram(variances, bins=var_bins)

print(hist)
print((bin_edges))
#
# olivetti = datasets.fetch_olivetti_faces()
# images = olivetti.images
#
# variances = np.empty((400, 64, 64))
# # print(variances)
#
# index = 0
# for img in images:
#     image_var = local_binary_pattern(np.round(img*255), 8, 1, 'var')
#     variances[index] = image_var
#     index += 1
#
# # print(np.isnan(np.sum(np.nan_to_num(variances))))
#
# # Convert NAN to 0.0
# variances = np.nan_to_num(variances)
#
#
# # This code is for finding the bin values for 16 bins for variance
# # Found to be: [0.0, 2.3420074916527938, 4.9597703118092795, 8.295748475028176, 12.555121737481386, 18.051872273956178, 25.16856667983984, 34.547447553750544, 47.03868615696774, 64.07498734477423, 88.00294140949154, 122.87955648325055, 176.1619694556398, 265.35077964955246, 443.3784170852141, 1057.9593746247647, 9722.143494012096]
# # # Extract 0th-16th quantile of the data
# # quantiles = []
# # for i in range(17):
# #     quantiles.append(1/16*i)
# #
# # var_bins = []
# # # Compute the bins for equally splitting the variance data
# # for q in quantiles:
# #     var_bins.append(np.quantile(variances, q))
# # print(var_bins)
#
# lbp_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# var_bins = [0.0, 2.3420074916527938, 4.9597703118092795, 8.295748475028176, 12.555121737481386, 18.051872273956178, 25.16856667983984, 34.547447553750544, 47.03868615696774, 64.07498734477423, 88.00294140949154, 122.87955648325055, 176.1619694556398, 265.35077964955246, 443.3784170852141, 1057.9593746247647, 9722.143494012096]
#
# # Construct histogram
# hist, bin_edges = np.histogram(variances, bins=var_bins)
#
# print(hist)
# print((bin_edges))
#
#
# # MAKE 2-D histogram with these values
