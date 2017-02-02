import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy import ndimage
import time


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
    grad =  ndimage.gaussian_gradient_magnitude(image, 5)
    avg_grad = np.average(grad)

    return grad, avg_grad
# GRADIENT ===================================================================================

# BACKGROUND_DECISION ========================================================================
def decision(img, grad, background_pixel, background_grad):
    foreground = img*1
    for z in range(len(img)):
        for s in range(len(img[1])):
            k = distance(img[z,s], grad[z,s], avg_grad, alpha, background_pixel[z,s], background_grad[z,s])
            if len(k[k<R_scale]) >= nmbr_min:
                foreground[z,s] = 0
            else:
                foreground[z,s] = 255    
    return foreground
# BACKGROUND_DECISION ========================================================================



# img = cv2.imread('D:/Benutzer/Chrisu/Desktop/temp/bilder/kinglet.jpg',0)

last_frame = 0

cap = cv2.VideoCapture('video_wecker.avi')

ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

grad, avg_grad = gradient(frame)

background_pixel = np.zeros((frame.shape[0],frame.shape[1],N))
for i in range(N):
    background_pixel[:,:,i] = frame

background_grad = np.zeros((grad.shape[0],grad.shape[1],N))
for i in range(N):
    background_grad[:,:,i] = grad

foreground = decision(frame,grad,background_pixel, background_grad)

while(cap.isOpened()):
    if ret == False:
        cap.release()
        cv2.destroyAllWindows()

    last_frame = frame
        
    foreground = decision(frame,grad,background_pixel, background_grad) 

    
    cv2.imshow('foreground', foreground)
    
    while not cv2.waitKey(1) & 0xFF == ord('p'):
        time.sleep(0.0001)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad, avg_grad = gradient(frame)
