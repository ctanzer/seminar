import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import cv2

img = cv2.imread('bild1.jpg')

rows, cols, chan = img.shape

img_small = cv2.resize(img, (cols/8, rows/8))

cv2.imwrite('bild1_small.jpg', img_small)
