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
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency

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
def main():
    model = load_model('kersa_1.h5')
    img = plt.imread("cropSampled/video_0099_8_crop.jpg")
    img_occ = img[:][:,:,0]
    img_occ.setflags(write=1)
    #img_occ[6:10,100:104] = 0
    #plt.imshow(img_occ,cmap='gray')
    #plt.show()
    X = img_occ.reshape(1, 160,160, 1)
    out = model.predict(X)[[0]]
    #print(out)
    # occlusion
    img_size = 160
    occlusion_size = 4
    label_99 = 2.544006547
    heatmap = np.zeros((img_size, img_size), np.float32)
    for n, (x, y, img_float) in enumerate(iter_occlusion(img, size=occlusion_size)):
        X = img_float[:,:,0].reshape(1, 160,160, 1)
        out = model.predict(X)[[0]]
        heatmap[y:y + occlusion_size, x:x + occlusion_size] = abs(out-label_99)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()




if __name__ == "__main__":
    main()
