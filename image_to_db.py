from pymongo import MongoClient
import sys
import connection_string, encode_image, numpy, ssl
import numpy as np

client = MongoClient(connection_string.connection_string(), ssl_cert_reqs=ssl.CERT_NONE)
db = client.cse515

# send image object to DB
def image_to_db(filepath, filename, model_name, fd):

    # encode the image
    encoded_image = encode_image.encode_image(f'{filepath}{filename}')

    # create face object with image string and id
    face = {
        'image': encoded_image,
        'id': filename,
        model_name: {
            'fd': fd,
        }
    }

    # insert into DB
    # replaces or inserts if doesn't exist
    result = db.phase2.replace_one({'id': filename}, face, True)

    # update how many images have been uploaded
    status = "Images: " + ' ' + format(result.upserted_id)
    print (status)

# retrieve image object from DB by id
def db_to_object(filename):
    return db.phase2.find_one({'id': filename})

def eigen_to_db(eigenpairs, filename):
    for pair in eigenpairs:
        pair[1] = pair[1].real.tolist()
    print(eigenpairs)

    eigen = {
        'id': filename,
        'eigenpairs': eigenpairs
    }

    result = db.phase2.replace_one({'id': filename}, eigen, True)

# Debugging
if __name__ == "__main__":
    count = 0
    img_encoded = db_to_object('image-0.png')
    img_decoded = encode_image.decode_image(img_encoded['image'])
    img_array = numpy.asarray(img_decoded)