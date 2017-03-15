import cv2
import numpy as np
from sklearn import datasets
from PIL import Image
from PIL import ImageChops

def imageProcessing(path):
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #   Grayscale
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)    # Threshold - simplest method of image segmentation. From a grayscale image, thresholding can be used to create binary images
    resized = cv2.resize(imgray,(8,8)) # 8 x 8 pixels
    cv2.imwrite('out.png', resized)
    image = resized
    return flatten(resized)

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

image = imageProcessing('1.png')
image2 = imageProcessing("2.png")
print image
print image2
#digits = datasets.load_digits()
#print digits.images[0]

