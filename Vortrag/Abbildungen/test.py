import numpy as np
import cv2


img = cv2.imread('R_lbsp.jpg')


while True:
    cv2.imshow('R',img*2)
    cv2.imwrite('R2_lbsp.jpg',img*2)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

