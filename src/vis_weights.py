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

model = load_model('trained_models/keras_r_80.h5')
#read image
#img = plt.imread(f"cropSampled/video_0031_10_crop.jpg")[:,:,0]
img = plt.imread(f"video_1418_10_pimp.jpg")[:,:,0]
X = img.reshape(1, 160,160, 1)
out = model.predict(X)[[0]]
print(out)
top_layer = model.layers[0].output
layer_outputs = [layer.output for layer in model.layers[:11]] 
print(np.shape(top_layer))
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X) 
first_layer_activation = activations[0]
print(first_layer_activation.shape)
layer_shape = first_layer_activation.shape[-1]
'''
fig, ax = plt.subplots(1,8)
for row in range(1):
    for col in range(8):
        ax[col].imshow(first_layer_activation[0, :, :, col],cmap='gray')
plt.show()
'''

for idx in range(layer_shape):
    plt.imshow(first_layer_activation[0, :, :, idx],cmap='gray')
    plt.show()

'''
print(np.shape(top_layer.get_weights()[0]))
print(top_layer.get_weights()[0])

print(np.shape(top_layer.get_weights()[0][:,:,:,:].squeeze()))
#print(top_layer.get_weights()[0][:,:,:,4].squeeze())
plt.imshow(top_layer.get_weights()[0][:, :, :, 4].squeeze(), cmap='gray')
#plt.show()
'''