# Test script from MongoDB
import ssl
from connection_string import connection_string
from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(connection_string(), ssl_cert_reqs=ssl.CERT_NONE)
db = client.admin
# Issue the serverStatus command and print the results
serverStatusResult=db.command("serverStatus")
pprint(serverStatusResult)

# Here is an example of getting an image from the DB.
# from image_to_db import db_to_image
# from encode_image import decode_image
# # Database connection
# client = MongoClient(connection_string.connection_string())
# db = client.cse515
# img_obj = db_to_image(image_id)
# im = decode_image(img_obj["image"])
