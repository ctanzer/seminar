import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

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

    r,c = image.shape

    # Attach a copy of the edges so, that the gradient is right at the edges of the origianal image
    image = np.concatenate((image,image[r-1:r,:]),axis=0)
    image = np.concatenate((image[0:1,:],image),axis=0)
    image = np.concatenate((image,image[:,c-1:c]),axis=1)
    image = np.concatenate((image[:,0:1],image),axis=1)

    grad_x, grad_y = np.gradient(image)

    print grad_x
    grad_x = grad_x[1:-1,1:-1]
    print grad_x

# GRADIENT ===================================================================================




img = cv2.imread('bild1_small.jpg',0)
print img

gradient(img)

B = img[0,0:len(img[0]):len(img[0])/N]*1

img_test = img*1
for i in range(len(img)):
    for j in range(len(img[0])):
        if distance(img[i,j], B, R_scale) >= nmbr_min:
            img_test[i,j] = 0;


plt.figure()
plt.imshow(img_test,cmap="gray")
plt.show()
