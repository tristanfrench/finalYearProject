import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import imsave
import numpy as np
import pandas as pd
import os
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.image as mpimg
import random
import matplotlib as mpl
from matplotlib.pyplot import figure
from skimage.transform import downscale_local_mean
from skimage import color


def image_preprocess():
    for f in listdir(os.getcwd()+'/'+'imageSample'+'/'):

        img = mpimg.imread('imageSample/'+f)[76:396,164:484]
        #img = color.rgb2gray(img)
        img = downscale_local_mean(img, (5,5))
        imsave('cropSampled2/'+f[:-4]+'_crop'+'.jpg',img,cmap=mpl.cm.gray)
        print(f)

 


#print(np.shape(mpimg.imread('imageSample/video_0001_8.jpg')))
#a=np.reshape(mpimg.imread('imageSample/video_0001_8.jpg'),[480,640,1])
#a=mpimg.imread('imageSample/video_0001_8.jpg')
'''
a=mpimg.imread('imageSample/video_0001_8.jpg')
a.setflags(write=1)
b=a[76:396,164:484]
print(np.shape(b))
b.setflags(write=1)
a[76,:]=255
a[396,:]=255

a[:,164]=255
a[:,484]=255
imshow(b,cmap='gray')
plt.show()
b = downscale_local_mean(b, (5,5))
print(np.shape(b))
imshow(b,cmap='gray')
plt.show()
#matplotlib.pyplot.imsave
'''
image_preprocess()


