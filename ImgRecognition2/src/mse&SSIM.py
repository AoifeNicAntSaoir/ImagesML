from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage.measure import structural_similarity, compare_ssim
import numpy as np



ima3 = np.array(Image.open("2ii.png").convert('L'))
img3= Image.open("2ii.png")
print img3.size[0]
if img3.size[0]>8:
    out = img3.thumbnail((8,8),Image.ANTIALIAS)
    #print img3.size[0]
ima4 = np.array(Image.open("2i.png").convert('L'))

print mean_squared_error(ima3, ima4)

print compare_ssim(ima3, ima4)