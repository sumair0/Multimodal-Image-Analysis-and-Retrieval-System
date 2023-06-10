import sys
from os import walk
from image_to_db import image_to_db

folder = sys.argv[1]

# upload all images to database
filenames = next(walk(folder))[2]
for filename in filenames:
    image_to_db(folder, filename)
