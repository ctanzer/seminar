import numpy as np
import cv2

img = cv2.imread('bild.png')
rows, cols, channels = img.shape

img1 = img[:,15:cols/2,:]
img2 = img[:,cols/2+15:cols,:]

img3 = img[2:rows,15:cols/2-17,:]
img4 = img[:rows-2,cols/2+32:cols,:]


while True:
    cv2.imshow('3',img3)
    cv2.imshow('4',img4)
    cv2.imshow('1+2', img1/2+img2/2)
    cv2.imshow('3+4', img3/2+img4/2)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
