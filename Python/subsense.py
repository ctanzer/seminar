import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

# Constants
T_r = 0.3
N_grid = 16
nmbr_min_grad = 12
MAX_hamming_weight = 0

N_back = 50
nmbr_min_back = 2
R = 10

T_update = 2

alpha = 0.03


# LBSP-UPDATE GRADIENT PICTURES ==========================================================
def update_grid_pictures(img, gradient_pictures, rows, cols):
    #    0    x   1   x   2
    #    x    3   4   5   x 
    #    6    7   x   8   9
    #    x   10  11  12   x
    #   13    x  14   x   15
    
    # First row
    gradient_pictures[:rows-2,:cols-2,0]  = img[2:rows,2:cols]
    gradient_pictures[:rows-2,:cols,1]    = img[2:rows,:cols]
    gradient_pictures[:rows-2,2:cols,2]   = img[2:rows,:cols-2]

    # Second row
    gradient_pictures[:rows-1,:cols-1,3]  = img[1:rows,1:cols]
    gradient_pictures[:rows-1,:cols,4]    = img[1:rows,:cols]
    gradient_pictures[:rows-1,1:cols,5]   = img[1:rows,:cols-1]

    # Third row
    gradient_pictures[:rows,:cols-2,6]    = img[:rows,2:cols]
    gradient_pictures[:rows,:cols-1,7]    = img[:rows,1:cols]
    gradient_pictures[:rows,1:cols,8]     = img[:rows,:cols-1]
    gradient_pictures[:rows,2:cols,9]     = img[:rows,:cols-2]

    # Fourth row
    gradient_pictures[1:rows,:cols-1,10]  = img[:rows-1,1:cols]
    gradient_pictures[1:rows,:cols,11]    = img[:rows-1,:cols]
    gradient_pictures[1:rows,1:cols,12]   = img[:rows-1,:cols-1]

    # Fifth row
    gradient_pictures[2:rows,:cols-2,13]  = img[:rows-2,2:cols]
    gradient_pictures[2:rows,:cols,14]    = img[:rows-2,:cols]
    gradient_pictures[2:rows,2:cols,15]   = img[:rows-2,:cols-2]

    return gradient_pictures
# LBSP-UPDATE GRADIENT PICTURES ==========================================================

# LBSP-GRADIENT DECISION =================================================================
def lbsp(img, gradient_pictures):
    global T_r#,d_min_new
    gradient_decision = gradient_pictures - img[:,:,np.newaxis]
    gradient = img*0
    T = T_r * img
    comp = gradient_decision<T[:,:,np.newaxis]
    gradient[(comp != False).sum(2) < nmbr_min_grad] = 255
   
    #d_min_new = np.absolute(np.amin(gradient_decision, axis=2)/255.0)
    # T_r update ------ Grenzen noch anpassen, eventuell auch Geschwindigkeit der Aenderung
    hammingweight=comp.sum()*1.0/MAX_hamming_weight
    if(T_r > 0.1 and T_r < 1):
        if(hammingweight < 0.865):
            T_r = T_r + 0.001
        if(hammingweight > 0.875):
            T_r = T_r - 0.001
        
    return gradient
# LBSP-GRADIENT DECISION =================================================================

# BACKGROUND DECISION ====================================================================
def decision(img,background_pictures):
    background = img*0
    comp = np.absolute(img[:,:,np.newaxis] - background_pictures) < R
    background[(comp != False).sum(2) <= nmbr_min_back] = 255;
    return background
# BACKGROUND DECISION ====================================================================

# BACKGROUND_UPDATE ======================================================================
def background_update(img, background, background_pictures):
    n = np.uint8(np.floor(N_back*np.random.random()))
    # Random pixels with probability 1/T
    rand_array = 100.*np.random.random(img.shape)
    update_array = np.logical_and(((100/T_update) > rand_array), background == 0)
    cv2.imshow('update_array', np.uint8(update_array)*255)
     # Update background pictures in plane n
    background_pictures[update_array,n] = img[update_array]
# BACKGROUND_UPDATE ======================================================================

# DISTANCE_UPDATE ========================================================================
def distance_update(img, background_pictures, rows, cols):
    global d_min_arr, d_min_new
    # Get the new minimum distance array
    d_min_new = np.amin((np.absolute(img[:,:,np.newaxis] - background_pictures)), axis = 2)/255.0
 
    d_min_arr = d_min_arr*(1-alpha) + d_min_new * alpha

    d_min_arr[d_min_arr < 0] = 0
    d_min_arr[d_min_arr > 1] = 1
# DISTANCE_UPDATE ========================================================================

# PROGRAM ######################################################################################
cap = cv2.VideoCapture('highway.avi')
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = img.shape

MAX_hamming_weight = rows*cols*N_grid

gradient_pictures = np.ones((rows,cols,N_grid))*255
update_grid_pictures(img, gradient_pictures, rows, cols)

d_min_new = img*255
d_min_arr = np.ones((rows, cols))

background_pictures = np.uint8(np.ones((rows, cols, N_back)) * img[:,:,np.newaxis])
background = decision(img, background_pictures)


while True:
    ret, img = cap.read()
    if ret == False:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    update_grid_pictures(img, gradient_pictures, rows, cols)
    gradient = lbsp(img, gradient_pictures)
    background_update(img, background, background_pictures)
    background = decision(img, background_pictures)
    distance_update(img, background_pictures, rows, cols)


    blured = cv2.medianBlur(background, 9)
    cv2.imshow('blured', blured)
    cv2.imshow('distance',np.uint8(d_min_arr*255))
    cv2.imshow('distance new',np.uint8(d_min_new*255))

    cv2.imshow('gradient',gradient)
    cv2.imshow('original',img)
    cv2.imshow('background',background)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# PROGRAM ######################################################################################


