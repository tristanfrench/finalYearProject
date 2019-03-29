import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
from keras import activations
from keras.callbacks import TensorBoard

import lime 
from lime import lime_image


import os
import os.path
from os import listdir
from os.path import isfile, join
import sys


explainer = lime_image.LimeImageExplainer()
model = load_model('trained_models/categ.h5')
#img = plt.imread("cropSampled/video_1858_9_crop.jpg")[:,:,0]
img = plt.imread("cropSampled/video_0684_11_crop.jpg")[:,:,0]
explanation = explainer.explain_instance(img, model.predict, top_labels=3, hide_color=0, num_samples=1000)
temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=5, hide_rest=False)
plt.imshow(mask)
plt.imshow(temp)
plt.show()