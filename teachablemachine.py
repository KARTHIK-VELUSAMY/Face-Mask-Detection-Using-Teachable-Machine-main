import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# clarity
np.set_printoptions(suppress=True)

# model
model = tensorflow.keras.models.load_model('E:/AI LAB/mask/teacable mask/keras_model.h5')

# Create the array 
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# image
image = Image.open('E:/AI LAB/mask/projects/face-mask-detector/1-with-mask.jpg')


#resizing 
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

# numpy array
image_array = np.asarray(image)


#  resized
image.show()

# Normalize
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array


# run the inference
prediction = model.predict(data)
print(prediction)


