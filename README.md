# Multimodal Image Analysis and Retrieval System
This project focuses on implementing a semantic image search and relevance feedback system using vector models, indexing and search techniques, classification algorithms, and relevance feedback mechanisms. Additionally, the project involves the implementation of Locality-Sensitive Hashing (LSH) and VA-Files for efficient image indexing and search operations.

Tasks within the project include implementing image labeling, latent semantics computation, classifier selection (SVM, decision-tree, PPR-based), false positive and miss rate computation, LSH tool development, similar image search using LSH index structure, VA-Files index tool development, similar image search using VA-Files index structure, decision-tree-based relevance feedback, SVM-classifier-based relevance feedback, and query and feedback interface implementation.

## Operating System

- Windows 10 Home Edition
- MacOS Big Sur 11.2.3

## Python Version

Python 3.9.4

## Required Modules

- OpenCV cv2
- NumPy
- Pandas
- Scikit-Image (scikit-image)
- Matplotlib
- OpenCV-Python
- Scikit-Learn (scikit-learn)
- CSVWriter
- Pillow
- Networkx

## Execution Instructions

1. Install all required modules by running the following commands in the command line:

```shell
pip install scikit-image
pip install matplotlib
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install csvwriter
pip install networkx
pip install pandas
```

2. Enter the desired task number from the options provided: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].

## Tasks

### Task 1

- Enter the path for the first folder.
- Enter the feature model (options: cm, elbp, hog).
- Enter the value of k.
- Enter the path for the second folder.
- Enter the classifier model to be used (options: SVM, decision tree, PPR).

Example:

```plaintext
Enter path for first folder:
1000
Enter feature model:
elbp
Enter k:
5
Enter path for second folder:
100
Enter classifier model to be used:
2
```

### Task 2

- Enter the path for the first folder.
- Enter the feature model (options: cm, elbp, hog).
- Enter the value of k.
- Enter the path for the second folder.
- Enter the classifier model to be used (options: SVM, decision tree, PPR).

Example:

```plaintext
Enter path for first folder:
1000
Enter feature model:
elbp
Enter k:
5
Enter path for second folder:
100
Enter classifier model to be used:
2
```

### Task 3

- Enter the path for the first folder.
- Enter the feature model (options: cm, elbp, hog).
- Enter the value of k.
- Enter the path for the second folder.
- Enter the classifier model to be used (options: SVM, decision tree, PPR).

Example:

```plaintext
Enter path for first folder:
1000
Enter feature model:
elbp
Enter k:
5
Enter path for second folder:
100
Enter classifier model to be used:
2
```

### Task 4

- Enter the filepath of the folder to perform LSH on.
- Enter the feature model (options: cm, elbp, hog).
- Enter the value of k.
- Enter the number of Layers.
- Enter the filepath of the image to perform search on.
- Enter the value of t.

Example:

```plaintext
Enter the filepath of the folder to perform LSH on:
100
Enter feature model:
elbp
Enter k value:
5
Enter number of Layers:
3
Enter filepath of image to perform search on:
test/image-cc-1-1.png
Enter t value:
