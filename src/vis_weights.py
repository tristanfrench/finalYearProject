import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

import keras
from keras import models
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
from keras import activations
from keras.callbacks import TensorBoard

import os
import os.path
from os import listdir
from os.path import isfile, join
import sys

model = load_model('trained_models/keras_r_40_second.h5')
#read image
#img = plt.imread('pimp_img/video_0456_10_mod.jpg')[:,:,0]
img = plt.imread(f"cropSampled/video_0456_10_crop.jpg")[:,:,0]

X = img.reshape(1, 160,160, 1)
out = model.predict(X)[[0]]
print(out)
current_layer = 0
top_layer = model.layers[current_layer].output
layer_outputs = [layer.output for layer in model.layers[:11]] 
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X) 
layer_activation = activations[current_layer]
layer_shape = layer_activation.shape[-1]
'''
fig, ax = plt.subplots(1,4)
ax[0].axis('off')
ax[0].imshow(img,cmap='gray') 
for row in range(1):
    for col in range(1,4):
        ax[col].axis('off')
        ax[col].imshow(layer_activation[0, :, :, col-1],cmap='gray') 
plt.show()
'''
#'''
fig, ax = plt.subplots(1,4)
ax[0].axis('off')
ax[0].imshow(img,cmap='gray') 
for idx in range(1,4):
    ax[idx].axis('off')
    ax[idx].imshow(layer_activation[0, :, :, idx],cmap='gray')
plt.show()
#'''
'''
plt.axis('off')
plt.imshow(first_layer_activation[0, :, :, idx],cmap='gray')
plt.imshow(first_layer_activation[0, :, :, idx],cmap='gray')
plt.imshow(first_layer_activation[0, :, :, idx],cmap='gray')
plt.show()
'''



'''
print(np.shape(top_layer.get_weights()[0]))
print(top_layer.get_weights()[0])

print(np.shape(top_layer.get_weights()[0][:,:,:,:].squeeze()))
#print(top_layer.get_weights()[0][:,:,:,4].squeeze())
plt.imshow(top_layer.get_weights()[0][:, :, :, 4].squeeze(), cmap='gray')
#plt.show()
'''