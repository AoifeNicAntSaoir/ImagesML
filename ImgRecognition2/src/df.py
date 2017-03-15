import numpy as np
import cv2
from resizeimage import resizeimage
from PIL import Image
import matplotlib.pyplot as plt
im = cv2.imread('testImage.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

resizeimage.resize_cover(imgray, [8,8])



ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
#cnt = contours[4]
#img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
cv2.imwrite(img,'out.png')