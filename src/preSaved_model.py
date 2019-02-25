import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
from keras import activations
from keras.callbacks import TensorBoard

import os
import os.path
from os import listdir
from os.path import isfile, join
import sys


#from vis.visualization import visualize_activation
#from vis.utils import utils

import pandas as pd
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
def single_square_occ(img, size=10):
    img_y,img_x = np.shape(img)
    nb_iter = int(img_y/size)
    colour_value = 0
    for row in range(0,nb_iter):
        for col in range(0,nb_iter):
            tmp = img.copy()
            y_0 = row*size
            y_1 = (row+1)*size 
            x_0 = col*size
            x_1 = (col+1)*size
            tmp[y_0:y_1,x_0:x_1] = colour_value

            yield tmp,y_0,y_1,x_0,x_1

def diag_occ(img, n_diag=16):
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

def better_diag_occ(img, n_diag=10):
    img_y,img_x = np.shape(img)
    line_angle = 3
    colour_value = 0
    special_row = 0

    ######################################################
    
    for i in range(0,img_y,n_diag):
        tmp = img.copy()
        row = 0
        col = i
        while tmp[row,col:col+n_diag].size: #check if array not empty
            tmp[row,col:col+n_diag] = colour_value
            row += 1
            col += line_angle
            if row>=160:
                break
        yield tmp,i,line_angle
    
    for i in range(0,int(np.floor(160/(n_diag/line_angle)))):
        tmp = img.copy() 
        col = 0
        row = special_row
        print(row)
        row_changed = 0
        alpha = 1
        beta = 0
        while tmp[row,col:col+alpha-beta].size: #check if array not empty
            tmp[row,col:col+alpha-beta] = colour_value
            row += 1
            if alpha-beta > n_diag:
                col += line_angle
                if row_changed == 0:
                    special_row = row
                    row_changed = 1
            else:
                beta -= line_angle
            if row>=160:
                break
        yield tmp,i,line_angle

       

def visualise_occlusion(img, model, occ_type, size):
    fig, ax = plt.subplots()
    ax.imshow(img,cmap='gray')
    img_size = 160
    heatmap = np.zeros((img_size, img_size), np.float32)
    original_prediction = model.predict(img.reshape(1, 160,160, 1))[0][0]
    if occ_type == 'square':
        for occ_img,y_0,y_1,x_0,x_1 in single_square_occ(img,size):
            x = occ_img.reshape(1, 160,160, 1)
            out = model.predict(x)[[0]]
            heatmap[y_0:y_1,x_0:x_1] = abs(out-original_prediction)
    elif occ_type == 'diag':
        for occ_img,it,i in diag_occ(img,size):
            x = occ_img.reshape(1, 160,160, 1)
            out = model.predict(x)[[0]]
            for j in range(size):
                if it == 0:
                    np.fill_diagonal(heatmap[i+j:],abs(out-original_prediction))
                else:
                    np.fill_diagonal(heatmap[:,i+j:],abs(out-original_prediction))
    elif occ_type == 'good_diag':
        for occ_img,it,i,line_angle in better_diag_occ(img,size):
            x = occ_img.reshape(1, 160,160, 1)
            out = model.predict(x)[[0]]
            if it == 0:
                row = 0
                col = i
            else:
                col = 0
                row = i
            while heatmap[row,col:col+size].size: #check if array not empty
                heatmap[row,col:col+size] = abs(out-original_prediction)
                row += 1
                col += line_angle
                if row>=160:
                    break
    ax.imshow(heatmap, alpha=0.6)
    print(original_prediction)
    print(np.max(heatmap),np.min(heatmap))
    

def show_line(dis, angle):
    point = [80,80] #centre of image
    dis = -dis*4 #convert mm in pixel
    angle = -angle*np.pi/180 #convert deg in rad
    angle_p = angle-90*np.pi/180
    x = 40 #length of red line
    perpen_point = [point[0]+dis*np.cos(angle_p),point[1]+dis*np.sin(angle_p)]
    new_point_1 = [perpen_point[0]+x*np.cos(angle),perpen_point[1]+x*np.sin(angle)]
    new_point_2 = [perpen_point[0]-x*np.cos(angle),perpen_point[1]-x*np.sin(angle)]
    return new_point_1,new_point_2

def get_labels(img_nb):
    #import labels and return r and theta for correct image
    img_nb = int(img_nb)
    labels = pd.read_csv("video_targets_minus1.csv")
    return labels['pose_1'][img_nb],labels['pose_6'][img_nb]

def main(argv):
    img_nb = argv[0]
    occlusion_type = argv[1]
    occ_size = int(argv[2])
    #import keras model
    model = load_model('trained_models/keras_angle_5.h5')
    #read image
    img = plt.imread(f"cropSampled/video_{img_nb}_10_crop.jpg")[:,:,0]
    for occ_img,i,line_angle in better_diag_occ(img,occ_size):
        plt.imshow(occ_img)
        plt.show()
    '''
    #occlusion algo
    visualise_occlusion(img,model,occlusion_type,occ_size)
    #show line that shows the edge
    r,theta = get_labels(img_nb)
    print(f'r is {r}, theta is {theta}')
    point_1,point_2 = show_line(r,theta)
    plt.plot([point_2[0],point_1[0]], [point_2[1],point_1[1]], 'r-')
    plt.show()
    '''






if __name__ == "__main__":
    main(sys.argv[1:])

#0031_10 size 20 good_diag black
#0032_10 size 5 square black 
#0041_10 size 5 good_diag black 