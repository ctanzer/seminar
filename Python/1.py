import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy import ndimage

# Constants
N = 35
nmbr_min = 2
R_inc_dec = 0.05
R_lower = 18
R_scale = 5
T_dec = 0.05
T_inc = 1
T_lower = 2
T_upper = 200

# DISTANCE ===================================================================================
def distance(pixel, B, R_scale):
    """ Returns an distant vector for the given pixel value and every value in vector B

    :pixel: an uint8 value, which represents a color code
    :B: background history
    :R_scale: Maximum distant value
    :returns: number of times the distance was smaller than the R_scale value

    """
    distan = 0

    # cast to float
    pixel = pixel*1.0

    # calculate the number of times the pixel color is nearer the background colors than the distance
    for i in B:
        if abs(pixel - i) < R_scale:
            distan = distan + 1

    return distan
# DISTANCE ===================================================================================

# GRADIENT ===================================================================================
def gradient(image):
    """ Returns the gradient in x and y direction of an picture

    :image: a grayscale image
    :returns: gradient in x and y direction

    """
    grad_x, grad_y = np.gradient(image)
    return grad_x, grad_y
# GRADIENT ===================================================================================




# img = cv2.imread('D:/Benutzer/Chrisu/Desktop/temp/bilder/kinglet.jpg',0)
img = cv2.imread('wecker.jpg',0)

# plt.figure()
# plt.imshow(img_test,cmap="gray")

plt.figure()
g = ndimage.gaussian_gradient_magnitude(img, 5)
plt.imshow(g, cmap='gray')
plt.show()
