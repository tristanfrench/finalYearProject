#import tensorflow as tf
import numpy as np
#import pandas as pd
import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
import os
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import math
from matplotlib.pyplot import imshow
from keras.callbacks import TensorBoard
#from vis.visualization import visualize_activation
#from vis.utils import utils
from keras import activations
#from vis.visualization import visualize_saliency

def iter_occlusion(image, size=8):
    # taken from https://www.kaggle.com/blargl/simple-occlusion-and-saliency-maps

    occlusion = np.full((size * 5, size * 5, 1), [0.5], np.float32)
    occlusion_center = np.full((size, size, 1), [0.5], np.float32)
    occlusion_padding = size * 2

    # print('padding...')
    image_padded = np.pad(image, ( \
    (occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0) \
    ), 'constant', constant_values = 0.0)

    for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):

        for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
            tmp = image_padded.copy()

            tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
                x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
                = occlusion

            tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center

            yield x - occlusion_padding, y - occlusion_padding, \
                tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]
def single_square_occ(img,size=10):
    img_y,img_x = np.shape(img)
    nb_iter = int(img_y/size)
    colour_value = 150
    for row in range(0,nb_iter):
        for col in range(0,nb_iter):
            tmp = img.copy()
            y_0 = row*size
            y_1 = (row+1)*size 
            x_0 = col*size
            x_1 = (col+1)*size
            tmp[y_0:y_1,x_0:x_1] = colour_value

            yield tmp,y_0,y_1,x_0,x_1



def main():
    model = load_model('kersa_1.h5')
    img = plt.imread("cropSampled/video_1608_10_crop.jpg")[:,:,0]

    #img_occ = img[:][:,:,0]
    #img_occ.setflags(write=1)
    #img_occ[0:50,:] = 0
    fig, ax = plt.subplots()
    ax.imshow(img,cmap='gray')
    #plt.show()
    #X = img_occ.reshape(1, 160,160, 1)
    #out = model.predict(X)[[0]]
    #print(out)
    # occlusion
    img_size = 160
    occlusion_size = 10
    label_99 = 2.544006547
    label_1608 = -0.152099177

    heatmap = np.zeros((img_size, img_size), np.float32)
    '''
    for n, (x, y, img_float) in enumerate(iter_occlusion(img, size=occlusion_size)):

    '''
    for occ_img,y_0,y_1,x_0,x_1 in single_square_occ(img):
        
        x = occ_img.reshape(1, 160,160, 1)
        out = model.predict(x)[[0]]
        heatmap[y_0:y_1,x_0:x_1] = abs(out-label_1608)
    ax.imshow(heatmap,alpha=0.6)
    #fig.colorbar(heatmap)
    plt.show()
    
    #print(heatmap)
    #print(heatmap[:2,::10])
    print(np.max(heatmap),np.min(heatmap))





if __name__ == "__main__":
    main()
