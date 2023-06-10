# 515Project
CSE 515: Multimedia and Web Databases Project Phase 2

This version is produced on our local machines, the settings are described below: 

Operating system: Windows 10 Home Edition; MacOS Big Sur 11.2.3
Python version: Python 3.9.4
Modules: OpenCV cv2, NumPy, Pandas, Sklearn, Skimage, CSVWriter, Pillow, Networkx

Execution instructions: 

In the command line, first install all required modules

pip install scikit-image
pip install matplotlib
pip install opencv-python
pip install numpy
pip install scikit-learn 
pip install csvwriter
pip install networkx
pip install pandas


To run the scripts:

Task 1

p2task1.py -m <CM/ELBP/HOG>, -x <{cc/con/detail/emboss/jitter/neg/noise1/noise2/original/poster/rot/smooth/stipple> -k <val> -d <PCA/SVD>'

Example: p2task1.py -m CM -x cc -k 2 -d PCA

Task 2

p2task2.py -m <CM/ELBP/HOG>, -y <{1 <= Y <= 40}> -k <val> -d <PCA/SVD>'

Example: p2task2.py -m CM -y 11 -k 2 -d PCA

Task 3

p2task3.py -m <CM/ELBP/HOG>, -k <val> -d <PCA/SVD>

Example: p2task3.py -m CM -k 2 -d PCA

Task 5

p2task5.py -q <query image file>, -l <latent semantics file> -n <number of similar images>

Example: p2task5.py -q all_images/image-cc-1-1.png, -l task12_PCA_CM_cc_2.csv -n 5

Task 8

p2task8.py -t <csv file>, -m <val> -n <val>

Example: p2task8.py -t task3_similarity_PCA_CM_2.csv, -m 3 -n 3
