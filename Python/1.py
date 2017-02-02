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
R_scale = 10
T_dec = 0.05
T_inc = 1
T_lower = 2
T_upper = 200
alpha = 1

# DISTANCE ===================================================================================
def distance(pixel, grad, avg_grad, alpha, background_pixel, background_grad):
    """ Returns an distance for the given pixel and gradient value

    :pixel: an uint8 value, which represents a color code
    :grad: gradient magnitude of the current pixel
    :avg_grad: average gradient of the last frame
    :alpha: constant
    :background_pixel,background_grad: background history values
    :returns: distance 
    
    """
    # calculate the distance
    return alpha/avg_grad*abs(grad-background_grad) + abs(pixel - background_pixel)

# DISTANCE ===================================================================================

# GRADIENT ===================================================================================
def gradient(image):
    """ Returns the gradient in x and y direction of an picture

    :image: a grayscale image
    :returns: gradient in x and y direction

    """
    grad =  ndimage.gaussian_gradient_magnitude(img, 5)
    avg_grad = np.average(grad)

    return grad, avg_grad
# GRADIENT ===================================================================================

# BACKGROUND_DECISION ========================================================================
def decision(img, grad, background_pixel, background_grad):
    foreground = img*1
    for z in range(len(img)):
        for s in range(len(img[1])):
            k = distance(img[z,s], grad[z,s], avg_grad, alpha, background_pixel[z,s], background_grad)
            if len(k[k<R_scale]) >= nmbr_min:
                foreground[z,s] = 0
            else:
                foreground[z,s] = 1    
    return foreground
# BACKGROUND_DECISION ========================================================================



# img = cv2.imread('D:/Benutzer/Chrisu/Desktop/temp/bilder/kinglet.jpg',0)
img = cv2.imread('wecker_small.jpg',0)
grad, avg_grad = gradient(img)



background_pixel = np.zeros((img.shape[0],img.shape[1],35))
background_grad = np.zeros((grad.shape[0],grad.shape[1],35))

for z in range(len(img)):
    for s in range(len(img[1])):
        k = distance(img[z,s], grad[z,s], avg_grad, alpha, background_pixel[z,s], background_grad)
        if len(k[k<R_scale]) >= nmbr_min:
            img[z,s] = 0


plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
