import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

# Constants
T_r = 0.3
grid = np.array([(1, 0, 1, 0, 2),(0, 1, 1, 1, 0),(1, 1, 0, 1, 1,),(0, 1, 1, 1, 0),(3, 0, 1, 0, 4)])
N = 16
nmbr_min = 12
MAX_hamming_weight = 0

# LBSP-UPDATE GRADIENT PICTURES ==========================================================
def update_gradient_pictures(img, gradient_pictures, rows, cols):
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

# LBSP-UPDATE GRADIENT PICTURES ==========================================================

# LBSP-GRADIENT DECISION =================================================================
def lbsp(img, gradient_pictures):
    global T_r
    gradient_decision = gradient_pictures - img[:,:,np.newaxis]
    gradient = img*0
    T = T_r * img
    comp = gradient_decision<T[:,:,np.newaxis]
    gradient[(comp != False).sum(2) < nmbr_min] = 255
    
    # T_r update ------ Grenzen noch anpassen, eventuell auch Geschwindigkeit der Aenderung
    hammingweight=comp.sum()*1.0/MAX_hamming_weight
    if(T_r > 0.1 and T_r < 1):
        if(hammingweight < 0.865):
            T_r = T_r + 0.001
        if(hammingweight > 0.875):
            T_r = T_r - 0.001
        
    return gradient
# LBSP-GRADIENT DECISION =================================================================


cap = cv2.VideoCapture('highway.avi')
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rows, cols = img.shape

MAX_hamming_weight = rows*cols*N

gradient_pictures = np.ones((rows,cols,N))*255;
update_gradient_pictures(img, gradient_pictures, rows, cols)

while True:
    ret, img = cap.read()
    if ret == False:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    update_gradient_pictures(img, gradient_pictures, rows, cols)
    gradient = lbsp(img, gradient_pictures)
    cv2.imshow('original',gradient)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
