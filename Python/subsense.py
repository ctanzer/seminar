import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import time

# Constants
T_r = 0.3
N_grid = 16
nmbr_min_grad = 12
MAX_hamming_weight = 0

N_back = 100
nmbr_min_back = 2
R = 10
# pan
#R_color =50
#normal
R_color = 30
R_lbsp = 3

T_update = 2
T_lower = 2
T_upper = 256

# pan
#alpha = 0.01
#normal
alpha = 0.03
v_incr = 1
v_decr = 0.1

downsample = 0.8

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
def decision(img,background_pictures, R_color_arr):
    background = img*0
    comp = np.absolute(img[:,:,np.newaxis] - background_pictures) < R_color_arr[:,:,np.newaxis]
    background[(comp != False).sum(2) <= nmbr_min_back] = 255;
    return background
# BACKGROUND DECISION ====================================================================

# BACKGROUND_UPDATE ======================================================================
def background_update(img, background, background_pictures, T_arr):
    n = np.uint8(np.floor(N_back*np.random.random()))
    # Random pixels with probability 1/T_arr
    rand_array = 100.*np.random.random(img.shape)
    update_array = np.logical_and(((100/T_arr) > rand_array), background == 0)
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
        update_array[new_x, new_y] = True
    # Update background pictures in plane n
    background_pictures[update_array,n] = img[update_array]
    cv2.imshow('update_array', np.uint8(update_array)*255)
# BACKGROUND_UPDATE ======================================================================

# DISTANCE_UPDATE ========================================================================
def distance_update(img, background_pictures, rows, cols):
    global d_min_arr, d_min_new
    # Get the new minimum distance array
    d_min_new = np.amin((np.absolute(img[:,:,np.newaxis] - background_pictures)), axis = 2)/255.0
 
    d_min_arr = d_min_arr*(1-alpha) + d_min_new * alpha
    # bound it from 0 to 1
    d_min_arr[d_min_arr < 0] = 0
    d_min_arr[d_min_arr > 1] = 1
# DISTANCE_UPDATE ========================================================================

# BLINKING PIXELS ========================================================================
def recognize_blinking_pixels(background,blured):
    global old_background, v_arr
    # Find blinking pixels
    blinking_pixels = (background ^ old_background)/255
    old_background = background
    # Increment or Decrement the v_arr, which shows how dynamic the region is
    v_arr[blinking_pixels == 1] = v_arr[blinking_pixels == 1] + v_incr
    v_arr[blinking_pixels == 0] = v_arr[blinking_pixels == 0] - v_decr
    # v_arr must be greater than zero and should be zero for foreground
    v_arr[v_arr < 0] = 0
    v_arr[blured == 255] = 0
    cv2.imshow('v',np.uint8(v_arr))    
# BLINKING PIXELS ========================================================================

# THRESHOLD UPDATE =======================================================================
def threshold_update(v_arr, d_min_arr):
    global R_arr, R_color_arr, R_lbsp_arr
    incr = (R_arr < np.square(1+2*d_min_arr))
    R_arr[incr] = R_arr[incr] + v_arr[incr]
    decr = (incr == False) & (v_arr != 0)
    R_arr[decr] = R_arr[decr] - (1/v_arr[decr])
    R_arr[R_arr < 1] = 1
    R_arr[(incr | decr) == False] = 1
    cv2.imshow('R', np.uint8(R_arr*10))

    R_color_arr = R_arr * R_color
    R_lbsp_arr = np.power(2, R_arr) + R_lbsp
    cv2.imshow('R color', np.uint8(R_color_arr))
    cv2.imshow('R lbsp', np.uint8(R_lbsp_arr))

# THRESHOLD UPDATE =======================================================================

# PROBABILITY_UPDATE =====================================================================
def probability_update(background, v_arr, d_min_arr):
    global T_arr
    incr = (v_arr != 0) & (d_min_arr != 0) & (background != 0)
    T_arr[incr] = T_arr[incr] + (1/(v_arr[incr] * d_min_arr[incr]))
    maxi = ((v_arr == 0) | (d_min_arr == 0)) & (background != 0)
    T_arr[maxi] = T_upper
    
    decr = (d_min_arr != 0) & (background == 0)
    T_arr[decr] = T_arr[decr] - (v_arr[decr]/d_min_arr[decr])
    mini = (d_min_arr == 0) & (background == 0)
    T_arr[mini] = T_lower

    T_arr[T_arr < 2] = T_lower
    T_arr[T_arr > 256] = T_upper

    cv2.imshow('T_arr', np.uint8(T_arr-1))

# PROBABILITY_UPDATE =====================================================================

# PROGRAM ######################################################################################
cap = cv2.VideoCapture('highway.avi')
print(cap.get(cv2.cv.CV_CAP_PROP_FPS))
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,None,fx=downsample, fy=downsample, interpolation = cv2.INTER_CUBIC)

rows, cols = img.shape

MAX_hamming_weight = rows*cols*N_grid

gradient_pictures = np.ones((rows,cols,N_grid))*255
update_grid_pictures(img, gradient_pictures, rows, cols)

d_min_new = img*255
d_min_arr = np.ones((rows, cols))

R_arr = np.ones((rows, cols))
R_color_arr = R_arr
R_lbsp_arr = R_arr

T_arr = np.ones((rows, cols)) * 2

v_arr = img*0.0

threshold_update(v_arr, d_min_arr)

background_pictures = np.uint8(np.ones((rows, cols, N_back)) * img[:,:,np.newaxis])
background = decision(img, background_pictures, R_color_arr)
old_background = background
blured = cv2.medianBlur(background, 9)



while True:
    # Start time
    start = time.time()

    ret, img = cap.read()
    if ret == False:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,None,fx=downsample, fy=downsample, interpolation = cv2.INTER_CUBIC)

    update_grid_pictures(img, gradient_pictures, rows, cols)
    gradient = lbsp(img, gradient_pictures)
    background_update(img, background, background_pictures, T_arr)
    background = decision(img, background_pictures, R_color_arr)
    blured = cv2.medianBlur(background, 5)
    
    distance_update(img, background_pictures, rows, cols)
    recognize_blinking_pixels(background,blured)
    threshold_update(v_arr, d_min_arr)
    probability_update(background, v_arr, d_min_arr)

    cv2.imshow('blured', blured)
    cv2.imshow('distance',np.uint8(d_min_arr*255))
    cv2.imshow('distance new',np.uint8(d_min_new*255))

    cv2.imshow('gradient',gradient)
    cv2.imshow('original',img)
    cv2.imshow('background',background)
   
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    # End time
    end = time.time()
    # Time elapsed
    seconds = end - start
    # Calculate frames per second
    fps  = 1 / seconds;
    print "Estimated frames per second : {0}".format(fps);
cap.release()
cv2.destroyAllWindows()
# PROGRAM ######################################################################################


