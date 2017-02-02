import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import cv2

img = cv2.imread('wecker.jpg')

rows, cols, chan = img.shape

img_small = cv2.resize(img, (cols/5, rows/5))

cv2.imwrite('wecker_small.jpg', img_small)
