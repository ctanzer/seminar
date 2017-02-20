import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
# from scipy import ndimage

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
    # grad =  ndimage.gaussian_gradient_magnitude(image, 3)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    grad = cv2.magnitude(grad_x, grad_y)
    avg_grad = np.average(grad)

    return grad, avg_grad
# GRADIENT ===================================================================================

# BACKGROUND_DECISION ========================================================================
def decision(img, grad, avg_grad, alpha, background_pixel, background_grad):
    global d_min

    foreback = img*0+255
    d = distance(img, grad, avg_grad, alpha, background_pixel, background_grad)
    d_min = np.amin(d, axis=2)
    comp = d<R_arr[:,:,np.newaxis]
    foreback[(comp != False).sum(2) > nmbr_min] = 0

    return foreback
# BACKGROUND_DECISION ========================================================================

# BACKGROUND_UPDATE_PROBABILITY ==============================================================
def background_probability(platzhalter):
    platzhalter = 1

    return platzhalter
# BACKGROUND_UPDATE_PROBABILITY ==============================================================

# BACKGROUND_UPDATE ==========================================================================
def background_update(img, foreback, background_pixel, background_grad, pixel_probabilities):
    # Random plane
    n = np.uint8(np.floor(35*np.random.random()))
    # Random pixels with probability 1/T
    rand_array = 100.*np.random.random(img.shape)
    update_array = np.logical_and((pixel_probabilities > rand_array), foreback == 0)
    # Update background pixels in plane n
    background_pixel[update_array,n] = img[update_array]
    grad, avg_grad = gradient(img)
    background_grad[update_array,n] = grad[update_array]
    # Update minimum distance array
    distance_update(n)
# BACKGROUND_UPDATE ==========================================================================

# DISTANCE_UPDATE ============================================================================
def distance_update(n):
    global d_min_arr, d_min, d_min_avg, N
    # Update minimum distance array
    d_min_arr[:,:,n] = d_min
    # Calculate average minimum distances
    d_min_avg = d_min_arr.sum(2)/N
# DISTANCE_UPDATE ============================================================================


# THRESHOLD_UPDATE ===========================================================================
def threshold_update():
    global R_arr

    th_update = R_arr > d_min_avg * R_scale

    R_arr[th_update] *= (1 - R_inc_dec)
    R_arr[~th_update] *= (1 + R_inc_dec)

# THRESHOLD_UPDATE ===========================================================================

# LEARNING_RATE_UPDATE ===========================================================================
def learn_update():
    global T_rate_arr

    update_inc = (foreback == 1) & (T_rate_arr < T_upper)
    update_dec = (foreback == 0) & (T_rate_arr > T_lower)
    T_rate_arr[update_inc] += T_inc/d_min[update_inc]
    T_rate_arr[update_dec] -= T_dec/d_min[update_dec]

# LEARNING_RATE_UPDATE ===========================================================================

cap = cv2.VideoCapture('video_small_converted.avi')
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pixel_probabilities = np.ones(img.shape) * 50

rows, cols = img.shape

foreback = np.zeros((rows,cols))

background_pixel = np.uint8(np.ones((rows, cols, N)) * img[:,:,np.newaxis])
grad, avg_grad = gradient(img)


background_grad = np.ones((rows,cols,N)) * grad.reshape((rows,cols,1))

d_min = np.zeros((rows, cols))
d_min_arr = np.zeros((rows, cols, N))
d_min_avg = np.zeros((rows, cols))

# R_arr = np.zeros((rows, cols))
R_arr = np.ones((rows, cols))*R_scale

T_rate_arr = np.ones((rows,cols))*100

foreback = decision(img, grad, avg_grad, alpha, background_pixel, background_grad)

while True:
    ret, img = cap.read()
    if ret == False:
        cap.release()
        cv2.destroyAllWindows()
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad, avg_grad2 = gradient(img)
    background_update(img, foreback ,background_pixel, background_grad, pixel_probabilities)
    threshold_update()
    foreback = decision(img, grad, avg_grad, alpha, background_pixel, background_grad)
    # foreback = cv2.medianBlur(foreback,23)


    cv2.imshow('foreground', foreback)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
