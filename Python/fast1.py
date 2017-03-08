import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

# Constants
N = 10
nmbr_min = 2
R_inc_dec = 0.05
R_lower = 18
R_scale = 5
T_dec = 0.1
T_inc = 1
T_lower = 2
T_upper = 150
alpha = 1

# Variable for checking initialization
init = 0

# DISTANCE =====================================================================
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

# DISTANCE =====================================================================

# GRADIENT =====================================================================
def gradient(image):
    """ Returns the gradient in x and y direction of an picture

    :image: a grayscale image
    :returns: gradient in x and y direction

    """
    grad_x, grad_y = np.gradient(image)
    grad = cv2.magnitude(grad_x, grad_y)
    avg_grad = np.average(grad)

    return grad, avg_grad
# GRADIENT =====================================================================

# BACKGROUND_DECISION ==========================================================
def decision(img, grad, avg_grad, alpha, background_pixel, background_grad):
    global d_min, R_arr

    foreback = img*0
    d = distance(img, grad, avg_grad, alpha, background_pixel, background_grad)
    cv2.imshow('distance', np.uint8((d[:,:,9]-d[:,:,9].min())/(d[:,:,9].max()-d[:,:,9].min())*255))
    d_min = np.amin(d, axis=2)
    comp = d<R_arr[:,:,np.newaxis]
    foreback[(comp != False).sum(2) < nmbr_min] = 255

    return foreback
# BACKGROUND_DECISION ==========================================================

# BACKGROUND_UPDATE ============================================================
def background_update(img, grad, avg_grad, foreback, background_pixel, background_grad, n):
    global N
    # Random pixels with probability 1/T
    rand_array = 100.*np.random.random(img.shape)
    update_array = np.logical_and(((100/T_rate_arr) > rand_array), foreback == 0)
    cv2.imshow('update_array', np.uint8(update_array)*255)
    # Choose adjacent pixels
    ind_x, ind_y = np.nonzero(update_array)
    rand_coords = [(-1,-1), (-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    # print 'loops: ', len(ind_x)
    for i in range(len(ind_x)):
        rand_x, rand_y = rand_coords[np.uint8(np.random.rand()*8)]
        new_x = ind_x[i]+rand_x
        new_y = ind_y[i]+rand_y
        if new_x < 0 or new_x >= update_array.shape[0]:
            new_x = ind_x[i]-rand_x
        if new_y < 0 or new_y >= update_array.shape[1]:
            new_y = ind_y[i]-rand_y
        update_array[new_x, new_y] = 1

    # Update background pixels in plane n
    background_pixel[update_array,n] = img[update_array]
    background_grad[update_array,n] = grad[update_array]
# BACKGROUND_UPDATE ============================================================

# DISTANCE_UPDATE ==============================================================
def distance_update(n):
    global d_min_arr, d_min, d_min_avg, N
    # Update minimum distance array
    d_min_arr[:,:,n] = d_min
    # Calculate average minimum distances
    d_min_avg = d_min_arr.sum(2)/N
# DISTANCE_UPDATE ==============================================================

# THRESHOLD_UPDATE =============================================================
def threshold_update():
    global R_arr

    th_update = R_arr > d_min_avg * R_scale

    R_arr[th_update] *= (1 - R_inc_dec)
    R_arr[~th_update] *= (1 + R_inc_dec)
    cv2.imshow('threshold', np.uint8((R_arr-R_arr.min())/(R_arr.max()-R_arr.min())*255))

# THRESHOLD_UPDATE =============================================================

# LEARNING_RATE_UPDATE =========================================================
def learn_update():
    global T_rate_arr

    update_inc = (foreback > 0)
    update_dec = (foreback == 0)
    T_rate_arr[update_inc & (d_min > 0)] += T_inc/d_min[update_inc & (d_min > 0)]
    T_rate_arr[update_dec & (d_min > 0)] -= T_dec/d_min[update_dec & (d_min > 0)]
    T_rate_arr[T_rate_arr < T_lower] = T_lower
    T_rate_arr[T_rate_arr > T_upper] = T_upper

# LEARNING_RATE_UPDATE =========================================================

# PROBABILITY_UPDATE ===========================================================
def probability_update():
    global pixel_probabilities, T_rate_arr, once

    pixel_probabilities = 1/T_rate_arr
# PROBABILITY_UPDATE ===========================================================

################################################################################
# PBAS - Main function =========================================================
def pbas(image):
    pbas.init
    if pbas.init == 0:
        pbas.init = 1

        pbas.rows, pbas.cols = image.shape
        pbas.foreback = np.zeros((rows,cols))
        pbas.grad, pbas.avg_grad = gradient(image)

        pbas.pixel_probabilities = np.ones(image.shape) * 50

        pbas.background_pixel = np.uint8(np.ones((rows, cols, N)) * img[:,:,np.newaxis])
        pbas.background_grad = np.ones((rows,cols,N)) * grad[:,:,np.newaxis]

        pbas.d_min = np.ones((rows, cols))
        pbas.d_min_arr = np.zeros((rows, cols, N))
        pbas.d_min_avg = np.zeros((rows, cols))

        # R_arr = np.zeros((rows, cols))
        pbas.R_arr = np.ones((rows, cols))*R_scale

        pbas.T_rate_arr = np.ones((rows,cols))*100

    foreback = decision(image, grad, avg_grad, alpha, background_pixel, background_grad)

    return foreback
# PBAS - Main function =========================================================
################################################################################

cap = cv2.VideoCapture('highway.avi')
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Random plane
        n = np.uint8(np.floor(N*np.random.random()))
        # Update minimum distance array
        distance_update(n)
        # Update decision threshold
        threshold_update()
        # Update learning rate
        learn_update()
        # Update pixel probability
        # probability_update()
        # Update background model
        background_update(img, grad, avg_grad, foreback ,background_pixel, background_grad, n)

        grad, avg_grad = gradient(img)
        foreback = decision(img, grad, avg_grad, alpha, background_pixel, background_grad)
        foreback = cv2.medianBlur(foreback,15)

        cv2.imshow('orig', img)
        cv2.imshow('foreground', foreback)
        # cv2.imshow('backgrad0', np.uint8(background_grad[:,:,0]))
        # cv2.imshow('backgrad5', np.uint8(background_grad[:,:,5]))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
