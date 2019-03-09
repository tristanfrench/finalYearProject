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

def single_square_occ(img, direction, size=10):
    #direction is not used here
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
def rect_occ(img, direction, size=10):
    #direction is not used here
    occ_half = 1
    img_y,img_x = np.shape(img)
    nb_iter = int(img_y/size)
    colour_value = 0
    for row in range(0,nb_iter):
        tmp = img.copy()
        y_0 = row*size
        y_1 = (row+1)*size 
        tmp[y_0:y_1,:] = colour_value
        yield tmp, occ_half
def my_switch(direction,row):
    if direction == 'left':
        return row >= 160
    else:
        return row <= 0

def diagonal_occ(img, direction, n_diag=10):
    img_y,img_x = np.shape(img)
    if direction == 'left':
        starting_row = 0
        delta_row = 1
    else:
        starting_row = img_y-1  #159
        delta_row = -1
    line_angle = 3
    colour_value = 0
    #first half
    occ_half = 0
    for i in range(0,img_y,n_diag):
        tmp = img.copy()
        row = starting_row
        col = i
        oscilator = 0
        while tmp[row,col:col+n_diag].size: #check if array not empty
            tmp[row,col:col+n_diag] = colour_value
            if np.mod(oscilator,2):
                col += line_angle
            else:
                col += 0
            row += delta_row
            oscilator += 1
            if my_switch(direction, row):
                break
        yield tmp, occ_half
    #second half
    occ_half = 1
    while True:
        tmp = img.copy() 
        col = 0
        done = 0
        row = starting_row
        row_changed = 0
        alpha = 1
        beta = 0
        oscilator = 0
        while tmp[row,col:col+alpha-beta].size: #check if array not empty
            tmp[row,col:col+alpha-beta] = colour_value
            row += delta_row
            if alpha-beta > n_diag:
                if np.mod(oscilator,2):
                    col += line_angle
                else:
                    col += 0
                if row_changed == 0:
                    starting_row = row
                    row_changed = 1
            else:
                if np.mod(oscilator,2):
                    beta -= line_angle
                else:
                    beta -= 0
                if my_switch(direction, row):
                    done = 1
            oscilator += 1
            if my_switch(direction, row):
                break
        yield tmp, occ_half

        if done == 1:
            break


def visualise_occlusion(img, model, occ_type, direction, occ_size):
    plt.imshow(img,cmap='gray')
    img_size = 160
    heatmap = np.zeros((img_size, img_size), np.float32)
    original_prediction = model.predict(img.reshape(1, 160,160, 1))[0][0]
    diag_values = []
    for occ_img, occ_half in occ_type(img, direction, occ_size):
        x = occ_img.reshape(1, 160,160, 1)
        out = model.predict(x)[[0]]
        diff = abs(out-original_prediction)
        #add to diag_values
        if occ_half == 0:
            diag_values = np.insert(diag_values,0,diff)
        else:
            diag_values = np.append(diag_values, diff)
        #add to heatmap
        row,col = np.where(occ_img==0)
        for it in range(len(col)):
            heatmap[row[it],col[it]] = diff
    plt.imshow(heatmap, alpha=0.6)
    #print(original_prediction)
    #print(np.max(heatmap),np.min(heatmap))
    #print(round(np.max(heatmap),3))
    return original_prediction, np.max(heatmap), np.min(heatmap), diag_values, heatmap
        
def occ_single_model(img, models, occ_type, direction, occ_size):
    
    for idx,model in enumerate(models):
        plt.imshow(img,cmap='gray')
        img_size = 160
        heatmap = np.zeros((img_size, img_size), np.float32)
        original_prediction = model.predict(img.reshape(1, 160,160, 1))[0][0]
        for occ_img in occ_type(img, direction, occ_size):
            x = occ_img.reshape(1, 160,160, 1)
            out = model.predict(x)[[0]]
            row,col = np.where(occ_img==0)
            for it in range(len(col)):
                heatmap[row[it],col[it]] = abs(out-original_prediction)
        plt.imshow(heatmap, alpha=0.6)
        print(original_prediction)
        print(np.max(heatmap),np.min(heatmap))
        print(round(np.max(heatmap),3))
        yield original_prediction, np.max(heatmap), np.min(heatmap)

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
    img_nb = int(img_nb)
    #import labels and return r and theta for correct image
    if img_nb == 10:
        raise Exception(f'The image number should not be 00{img_nb} to avoid confusion.')
    if img_nb < 10:
        img_nb = img_nb - 1 #first video is 0001 not 0000
    elif img_nb > 10:
        img_nb = img_nb - 2
    labels = pd.read_csv("video_targets_minus1.csv")
    return labels['pose_1'][img_nb],labels['pose_6'][img_nb]

def find_occ_type(s,theta):
    if s == 'square':
        output = single_square_occ
    elif s == 'rect':
        output = rect_occ
    elif s == 'diag':
        output = diagonal_occ
    return output

def get_midpoint(a,b):
    '''
    Return y first then x because x is col and y is row
    '''
    return [round((a[1]+b[1])/2),round((a[0]+b[0])/2)]
def get_accuracy(heatmap, diag_values, mid_point):
    top_3 = np.sort(diag_values)[-3:]
    mid_point_in_top3 = False
    for n in top_3:
        row,col = np.where(heatmap==n)
        for i in range(len(col)):
            if row[i]==mid_point[0] and col[i]==mid_point[1]:
                mid_point_in_top3 = True
    return mid_point_in_top3
        

def main(argv):
    img_nb = argv[0]
    occ_size = int(argv[1])
    r,theta = get_labels(img_nb)
    if theta < -15:
        direction = 'left'
        occ_type = diagonal_occ
    elif theta > 15:
        direction = 'right'
        occ_type = diagonal_occ
    else:
        direction = 0
        occ_type = rect_occ
    #find optimal occlusion algorithm
    #occ_type = find_occ_type(argv[1], theta)
    #import keras model
    #models = [load_model('trained_models/keras_angle_5.h5'),load_model('trained_models/keras_angle_40.h5'),load_model('trained_models/keras_angle_80.h5')]
    #models = [load_model('trained_models/keras_r_10.h5'),load_model('trained_models/keras_r_40.h5'),load_model('trained_models/keras_r_80.h5')]   
    models = load_model('trained_models/keras_angle_5.h5')
    #read image
    img = plt.imread(f"cropSampled/video_{img_nb}_10_crop.jpg")[:,:,0]

    '''
    for occ_img in diagonal_occ(img, direction, occ_size):
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img,cmap='gray')
        ax[1].imshow(occ_img,cmap='gray')
        plt.show()
    '''
    print(f'r is {r}, theta is {theta}')
    #occlusion algo
    
    original_prediction, max_diff, min_diff, diag_values, heatmap = visualise_occlusion(img, models, occ_type, direction, occ_size)
    #show line that shows the edge
    point_1,point_2 = show_line(r, theta)
    mid_point = get_midpoint(point_1,point_2)
    plt.plot([point_2[0],point_1[0]], [point_2[1],point_1[1]], 'r-')
    print(diag_values)
    print(point_1,point_2,mid_point)
    print('acc=',get_accuracy(heatmap, diag_values, mid_point))
    plt.show()

    
    '''
    for original_prediction, max_diff, min_diff in occ_single_model(img, models, occ_type, direction, occ_size):
        #show line that shows the edge
        point_1,point_2 = show_line(r, theta)
        plt.plot([point_2[0],point_1[0]], [point_2[1],point_1[1]], 'r-')
    
    plt.show()
    '''
    
    






if __name__ == "__main__":
    main(sys.argv[1:])

#0031_10 size 20 good_diag black
#0032_10 size 5 square black 
#0041_10 size 5 good_diag black 

#0044 good_diag 10