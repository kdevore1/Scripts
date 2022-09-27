# The purpose of this script is to import an image, convert the image to a 2D array, flatten the 2D array to a 1D array, then backproject the 1D array to 2D.


# Libraries
import numpy as np
from PIL import Image
from skimage import color
from skimage import io
import matplotlib.pyplot as plt


# Import your image
img = Image.open("insert_image_here")

# Displays image
plt.imshow(img)

# Print the shape of the array 
np.shape(img)


# Convert image to grayscale if necessary
# img_gray = color.rgb2gray(img)

# Displays grayscale image
# plt.imshow(img_gray)

# Convert image to a numpy array
np_array_img = np.array(img_gray) # if already grayscale, change file name here

# Print the shape of the array [rows, columns]
np.shape(np_array_img)


# Flatten 2D array to 1D 
flattened_1d_array = np_array_img.flatten()
print(flattened_1d_array)


# Reshape 1D array to 2D array
backprojection = np.reshape(flattened_1d_array, (854,1000)) # will have to change the (columns, rows) based on size original size of image
print(backprojection)


# Confirm array is 2D
np.shape(backprojection)


# Display reshaped array as image
plt.imshow(backprojection, interpolation = 'antialiased')
plt.show()
