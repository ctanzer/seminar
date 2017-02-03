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
R_scale = 15
T_dec = 0.05
T_inc = 1
T_lower = 2
T_upper = 200
alpha = 1

# DISTANCE ===================================================================================
def distance(img, grad, avg_grad, alpha, background_pixel, background_grad):
    """ Returns an distance for the given pixel and gradient value

    :pixel: an uint8 value, which represents a color code
    :grad: gradient magnitude of the current pixel
    :avg_grad: average gradient of the last frame
    :alpha: constant
    :background_pixel,background_grad: background history values
    :returns: distance

    """
    # calculate the distance
    return alpha/avg_grad*abs(grad[:,:,np.newaxis]-background_grad) + abs(img[:,:,np.newaxis] - background_pixel)

# DISTANCE ===================================================================================

# GRADIENT ===================================================================================
def gradient(image):
    """ Returns the gradient in x and y direction of an picture

    :image: a grayscale image
    :returns: gradient in x and y direction

    """
    grad =  ndimage.gaussian_gradient_magnitude(image, 3)
    avg_grad = np.average(grad)

    return grad, avg_grad
# GRADIENT ===================================================================================

# BACKGROUND_DECISION ========================================================================
def decision(img, grad, avg_grad, alpha, background_pixel, background_grad):
    foreback = img*0+255
    d = distance(img, grad, avg_grad, alpha, background_pixel, background_grad)
    comp = d<R_scale
    foreback[(comp != False).sum(2) > nmbr_min] = 0

    return foreback
# BACKGROUND_DECISION ========================================================================

# BACKGROUND_UPDATE_PROBABILITY ==============================================================
def background_probability(platzhalter):
    platzhalter = 1

    return platzhalter
# BACKGROUND_UPDATE_PROBABILITY ==============================================================

# BACKGROUND_UPDATE ==========================================================================
def background_update(img, foreback, background_pixel, background_grad):
    n = np.uint8(np.floor(35*np.random.random()))
    background_pixel[foreback == 0,n] = img[foreback == 0]    
    grad, avg_grad = gradient(img)
    background_grad[foreback == 0,n] = grad[foreback == 0]
# BACKGROUND_UPDATE ==========================================================================


cap = cv2.VideoCapture('video_small.avi')
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.imread('wecker_small.jpg',0)

rows, cols = img.shape

foreback = np.zeros((rows,cols))

background_pixel = np.uint8(np.ones((rows, cols, N)) * img[:,:,np.newaxis])
grad, avg_grad = gradient(img)


background_grad = np.ones((rows,cols,N)) * grad.reshape((rows,cols,1))


foreback = decision(img, grad, avg_grad, alpha, background_pixel, background_grad)

while True:
    ret, img = cap.read()
    if ret == False:
        cap.release()
        cv2.destroyAllWindows()
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad, avg_grad2 = gradient(img)
    background_update(img, foreback ,background_pixel, background_grad)
    foreback = decision(img, grad, avg_grad, alpha, background_pixel, background_grad)
    foreback = cv2.medianBlur(foreback,23)

    
    cv2.imshow('foreground', foreback)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
