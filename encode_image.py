import base64, io
from PIL import Image

# encode the image as base64
def encode_image(file):
    with open(file, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read())
    return encoded_image

# decode the image as base64
def decode_image(encoded_image):
    decoded_image = base64.b64decode(encoded_image)
    img = Image.open(io.BytesIO(decoded_image))
    return img
