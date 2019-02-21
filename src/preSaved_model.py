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
    colour_value = 255
    for it in range(2):
        for i in range(0,img_y,n_diag):
            tmp = img.copy()
            for j in range(n_diag):
                if it == 0:
                    np.fill_diagonal(tmp[i+j:],colour_value)
                else:
                    np.fill_diagonal(tmp[:,i+j:],colour_value)
            yield tmp,it,i

def better_diag_occ(img,n_diag=10):
    img_y,img_x = np.shape(img)
    line_angle = 2
    colour_value = 0
    for it in range(2):
        for i in range(0,img_y,n_diag):
            tmp = img.copy()
            if it == 0:
                row = 0
                col = i
            else:
                col = 0
                row = i
            print(row)
           
            while tmp[row,col:col+n_diag].size: #check if array not empty
                tmp[row,col:col+n_diag] = colour_value
                row += 1
                col += line_angle
                if row>=160:
                    break
            yield tmp,it,i
       



def visualise_occlusion(model,occ_type,size):
    img = plt.imread("cropSampled/video_1007_14_crop.jpg")[:,:,0]
    fig, ax = plt.subplots()
    ax.imshow(img,cmap='gray')
    img_size = 160
    heatmap = np.zeros((img_size, img_size), np.float32)
    original_prediction = model.predict(img.reshape(1, 160,160, 1))[[0]]
    if occ_type == 'square':
        for occ_img,y_0,y_1,x_0,x_1 in single_square_occ(img,size):
            x = occ_img.reshape(1, 160,160, 1)
            out = model.predict(x)[[0]]
            heatmap[y_0:y_1,x_0:x_1] = abs(out-original_prediction)
        #fig.colorbar(heatmap)
    elif occ_type == 'diag':
        for occ_img,it,i in diag_occ(img,size):
            x = occ_img.reshape(1, 160,160, 1)
            out = model.predict(x)[[0]]
            for j in range(size):
                if it == 0:
                    np.fill_diagonal(heatmap[i+j:],abs(out-original_prediction))
                else:
                    np.fill_diagonal(heatmap[:,i+j:],abs(out-original_prediction))
    ax.imshow(heatmap,alpha=0.6)
    print(original_prediction)
    print(np.max(heatmap),np.min(heatmap))
    plt.show()


def main():
    model = load_model('trained_models/keras_angle_5.h5')

    #video_0002_10_crop
    #video_0001_11_crop
    #video_0011_11_crop
    #init: video_1007_14_crop
    '''
    for new_img in diag_occ(img):
        plt.imshow(new_img,cmap='gray')
        plt.show()
    '''
    #img_occ = img[:][:,:,0]
    #img_occ.setflags(write=1)
    #img_occ[0:50,:] = 0

    #plt.show()
    #X = img_occ.reshape(1, 160,160, 1)
    #out = model.predict(X)[[0]]
    #print(out)
    # occlusion
    occlusion_size = 10
    n_diag = 40
    #visualise_occlusion(model,'diag',20)
    img = plt.imread("cropSampled/video_1007_14_crop.jpg")[:,:,0]
    for occ_img,_,_ in better_diag_occ(img):
        plt.imshow(occ_img)
        plt.show()
    label_99 = 2.544006547
    label_1608 = -0.152099177
    label_1 = -1.26
    label_2 = 8.14
    label_11 = 4.132


    





if __name__ == "__main__":
    main()
