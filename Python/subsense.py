import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

# Constants
T_d = 5
grid = np.array([(1, 0, 1, 0, 1),(0, 1, 1, 1, 0),(1, 1, 0, 1, 1,),(0, 1, 1, 1, 0),(1, 0, 1, 0, 1)])
print grid
# LBSP-BACKGROUND DECISION =================================================================
def lbsp(img):

    return 
# LBSP-BACKGROUND DECISION =================================================================


cap = cv2.VideoCapture('highway.avi')
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


while True:
    ret, img = cap.read()
    if ret == False:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('original', img)
    #cv2.imshow('foreground', foreback)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
