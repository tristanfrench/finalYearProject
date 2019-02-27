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
            yield tmp

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
            yield tmp

def left_diag_occ(img, n_diag=10):
    img_y,img_x = np.shape(img)
    line_angle = 3
    colour_value = 0
    special_row = 0
    for i in range(0,img_y,n_diag):
        tmp = img.copy()
        row = 0
        col = i
        delta_switch = 0
        while tmp[row,col:col+n_diag].size: #check if array not empty
            tmp[row,col:col+n_diag] = colour_value
            if np.mod(delta_switch,2):
                col += line_angle
            else:
                col += 0
            row += 1
            delta_switch += 1
            if row >= 160:
                break
        yield tmp
    for i in range(0,int(np.ceil(160/(n_diag/(line_angle/2))))-1):
        tmp = img.copy() 
        col = 0
        row = special_row
        row_changed = 0
        alpha = 1
        beta = 0
        delta_switch = 0
        if special_row == 160:
            break
        while tmp[row,col:col+alpha-beta].size: #check if array not empty
            tmp[row,col:col+alpha-beta] = colour_value
            row += 1
            if alpha-beta > n_diag:
                if np.mod(delta_switch,2):
                    col += line_angle
                else:
                    col += 0
                if row_changed == 0:
                    special_row = row
                    row_changed = 1
            else:
                if np.mod(delta_switch,2):
                    beta -= line_angle
                else:
                    beta -= 0
            delta_switch += 1
            if row >= 160:
                break
        yield tmp

def right_diag_occ(img, n_diag=10):
    img_y,img_x = np.shape(img)
    line_angle = 3
    colour_value = 0
    special_row = 0
    for i in range(0,img_y,n_diag):
        tmp = img.copy()
        row = img_y-1  #159
        col = i
        delta_switch = 0
        while tmp[row,col:col+n_diag].size: #check if array not empty
            tmp[row,col:col+n_diag] = colour_value
            if np.mod(delta_switch,2):
                col += line_angle
            else:
                col += 0
            row -= 1
            delta_switch += 1
            if row <= 0:
                break
        yield tmp
    special_row = img_y-1
    for i in range(0,int(np.ceil(160/(n_diag/(line_angle/2))))-1):
        tmp = img.copy() 
        col = 0
        row = special_row
        row_changed = 0
        alpha = 1
        beta = 0
        delta_switch = 0
        if special_row == 0:
            break
        while tmp[row,col:col+alpha-beta].size: #check if array not empty
            tmp[row,col:col+alpha-beta] = colour_value
            row -= 1
            if alpha-beta > n_diag:
                if np.mod(delta_switch,2):
                    col += line_angle
                else:
                    col += 0
                if row_changed == 0:
                    special_row = row
                    row_changed = 1
            else:
                if np.mod(delta_switch,2):
                    beta -= line_angle
                else:
                    beta -= 0
            delta_switch += 1
            if row <= 0:
                break
        yield tmp

def visualise_occlusion(img,model, occ_type, occ_size):
    fig, ax = plt.subplots()
    ax.imshow(img,cmap='gray')
    img_size = 160
    heatmap = np.zeros((img_size, img_size), np.float32)
    original_prediction = model.predict(img.reshape(1, 160,160, 1))[0][0]
    for occ_img in occ_type(img,occ_size):
        x = occ_img.reshape(1, 160,160, 1)
        out = model.predict(x)[[0]]
        row,col = np.where(occ_img==0)
        for it in range(len(col)):
            heatmap[row[it],col[it]] = abs(out-original_prediction)
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

def find_occ_type(s,theta):
    if s == 'square':
        output = single_square_occ
    elif s == 'good_diag':
        if theta < 0:
            output = left_diag_occ
        else:
            output = right_diag_occ
    return output

def main(argv):
    img_nb = argv[0]
    occ_size = int(argv[2])
    r,theta = get_labels(img_nb)
    #find optimal occlusion algorithm
    occ_type = find_occ_type(argv[1], theta)
    #import keras model
    model = load_model('trained_models/keras_angle_5.h5')
    #read image
    img = plt.imread(f"cropSampled/video_{img_nb}_10_crop.jpg")[:,:,0]
    '''
    to_show = np.zeros([160,160])
    for occ_img,it,i,line_angle in right_diag_occ(img,occ_size):
        row,col = np.where(occ_img==0)
        plt.imshow(to_show)
        plt.show()
    '''
    #occlusion algo
    visualise_occlusion(img,model,occ_type,occ_size)
    print(f'r is {r}, theta is {theta}')
    #show line that shows the edge
    point_1,point_2 = show_line(r,theta)
    plt.plot([point_2[0],point_1[0]], [point_2[1],point_1[1]], 'r-')
    plt.show()
    
    






if __name__ == "__main__":
    main(sys.argv[1:])

#0031_10 size 20 good_diag black
#0032_10 size 5 square black 
#0041_10 size 5 good_diag black 

#0044 good_diag 10