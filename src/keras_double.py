import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import os
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from matplotlib.pyplot import imshow
from keras.callbacks import TensorBoard
from keras import activations
import sys


#hyperparameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 64, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_integer('max_epochs', 1,'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('normalize', 1,'Normalise data or not(default: %(default)d)')
class Label():
    def __init__(self, x):
        self.x = x
        self.x_min = np.min(x)
        self.x_max = np.max(x)

    def norm(self):
        '''
        Normalize whole array, min max normalization
        '''
        return [(i-self.x_min)/(self.x_max-self.x_min) for i in self.x]
    def __repr__(self):
        return self.x

def inverse_norm(self, x, x_max, x_min):
    return [i*(x_max-x_min)+x_min for i in x]
def image_preprocess(image_dir, label_dir):
    '''
    Returns image names and labels.
    Image names returned are strings with name of image folder such as: 'cropSampled/video_0001_10_crop.jpg'.
    '''
    #load labels
    ground_truth = pd.read_csv("video_targets_minus1.csv")
    r = ground_truth['pose_1'].values
    theta = ground_truth['pose_6'].values
    #instantiate class
    r_class = Label(r)
    theta_class = Label(theta)
    #normalize data
    if FLAGS.normalize:
        r = r_class.norm
        theta = theta_class.norm
    #define matrix containing angle and dislacement as columns
    labels = np.zeros([len(r),2])
    labels[:,0] = r
    labels[:,1] = theta
    #duplicate each label 5 times because we use 5 images from each video which all have the same label
    n_duplicates = 5
    labels=[i for i in labels for _ in np.arange(n_duplicates)] 
    #images
    all_image_names = [image_dir+img_name for img_name in listdir(os.getcwd()+'/'+image_dir)]
    return all_image_names, labels

def img_generator(filenames, labels, batch_size):
    '''
    Returns batches of (images,labels)
    '''
    L = len(filenames)
    #this line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = np.array([mpimg.imread(img)[:,:,0] for img in filenames[batch_start:limit]]).reshape(-1,160,160,1)
            Y = np.array(labels[batch_start:limit])
            yield (X,Y)   
            batch_start += batch_size   
            batch_end += batch_size

def main(argv):
    #reset main graph
    tf.keras.backend.clear_session()
    #Load all data
    image_dir = 'cropSampled/'
    label_dir = 'video_targets_minus1.csv'
    images, labels = image_preprocess(image_dir, label_dir)
    #shuffle all data, labels and images shuffled in same way by keeping same seed
    rng = np.random.randint(1000)
    np.random.seed(rng)
    np.random.shuffle(images)
    np.random.seed(rng)
    np.random.shuffle(labels)
    #train/test/validation set
    train_data_images = images[:7500]
    train_data_labels = labels[:7500]
    val_data_images = images[7500:8750]
    val_data_labels = labels[7500:8750]
    test_data_images = images[8750:]
    test_data_labels = labels[8750:]
 
    #model architecture
    model = keras.Sequential()
    #Conv1
    model.add(Conv2D(8, kernel_size=5, padding='SAME', activation='relu',input_shape=(160,160,1),name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
    #Conv2
    model.add(Conv2D(16, kernel_size=5, padding='SAME', activation='relu') )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
    #Conv3
    model.add(Conv2D(32, kernel_size=5, padding='SAME', activation='relu') )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
    #Conv4
    model.add(Conv2D(32, kernel_size=5, padding='SAME', activation='relu') )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
    #Flatten
    model.add(Flatten())
    #Dense1
    model.add(Dense(800, activation='relu'))
    #Dense2
    model.add(Dense(2,name="preds"))
    #optimizer and loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    #create generators
    train_generator = img_generator(train_data_images,train_data_labels,FLAGS.batch_size)
    val_generator = img_generator(val_data_images,val_data_labels,FLAGS.batch_size)
    test_generator = img_generator(test_data_images,test_data_labels,FLAGS.batch_size)
    #define logs directory for tensorboard
    tensorboard = TensorBoard(log_dir="logs/keras_runs")
    #define steps
    steps_per_epoch = math.ceil(len(train_data_images)/FLAGS.batch_size)
    val_steps = math.ceil(len(val_data_images)/FLAGS.batch_size)
    #Training
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=FLAGS.max_epochs, validation_data=val_generator, validation_steps=val_steps, verbose=1, callbacks=[tensorboard])
    #Evaluation
    test_steps = len(test_data_images)/FLAGS.batch_size
    print(model.evaluate_generator(test_generator, steps=test_steps))
    print(model.metrics_names)
    
    #model.save(f'trained_models/kerasdouble_{argv[0]}_{FLAGS.max_epochs}_second.h5')
    
    #check a prediction:
    img_to_see = plt.imread("cropSampled/video_1794_8_crop.jpg")[:,:,0]
    X = img_to_see.reshape(1, 160,160, 1)
    out = model.predict(X)[0]
    inverse_norm(out,)
    print(out) 
    print(FLAGS.normalize,'norm')
    
    




if __name__ == '__main__':
    main(sys.argv[1:])
