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

def diag_occ(img,n_diag=16):
    img_y,img_x = np.shape(img)
    nb_iter = int(img_y/n_diag)
    colour_value = 0
    for it in range(2):
        for i in range(0,img_y,n_diag):
            tmp = img.copy()
            for j in range(n_diag):
                if it == 0:
                    np.fill_diagonal(tmp[i+j:],colour_value)
                else:
                    np.fill_diagonal(tmp[:,i+j:],colour_value)
            yield tmp,it,i


def main():
    model = load_model('kersa_1.h5')
    img = plt.imread("cropSampled/video_0011_11_crop.jpg")[:,:,0]
    #video_0002_10_crop
    #video_0001_11_crop
    #video_0011_11_crop
    '''
    for new_img in diag_occ(img):
        plt.imshow(new_img,cmap='gray')
        plt.show()
    '''
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
    n_diag = 20

    label_99 = 2.544006547
    label_1608 = -0.152099177
    label_1 = -1.26
    label_2 = 8.14
    label_11 = 4.132


    heatmap = np.zeros((img_size, img_size), np.float32)
    '''
    for occ_img,y_0,y_1,x_0,x_1 in single_square_occ(img):
        
        x = occ_img.reshape(1, 160,160, 1)
        out = model.predict(x)[[0]]
        heatmap[y_0:y_1,x_0:x_1] = abs(out-label_1608)
    ax.imshow(heatmap,alpha=0.6)
    #fig.colorbar(heatmap)
    plt.show()
    '''
   
    for occ_img,it,i in diag_occ(img,n_diag):
        
        x = occ_img.reshape(1, 160,160, 1)
        out = model.predict(x)[[0]]
        for j in range(n_diag):
            if it == 0:
                np.fill_diagonal(heatmap[i+j:],abs(out-label_11))
            else:
                np.fill_diagonal(heatmap[:,i+j:],abs(out-label_11))
    ax.imshow(heatmap,alpha=0.6)
    print(np.max(heatmap),np.min(heatmap))
    plt.show()
    





if __name__ == "__main__":
    main()
