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
tf.app.flags.DEFINE_integer('max_epochs', 10,'Number of mini-batches to train on. (default: %(default)d)')

class ImageLabelGenerator(object):
    def __init__(self, image_dir, label_dir, feature):
        #attributes
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.feature = feature
        self.image_names = None
        self.labels = None
        self.train_set = {}
        self.val_set = {}
        self.test_set = {}
        self.batch_size = FLAGS.batch_size
        #setup
        self.__data_setup()
    def __data_setup(self):
        '''
        Defines image names and labels.
        Image names are strings with name of image folder such as: 'cropSampled/video_0001_10_crop.jpg'.
        Defines all data sets
        '''
        #labels
        labels = pd.read_csv(self.label_dir)
        if self.feature == 'r':
            labels = labels['pose_1'].values
        elif self.feature == 'theta':
            labels = labels['pose_6'].values
        else:
            raise Exception('Invalid Argument. Only r and theta are valid arguments')
        #duplicate each label 5 times because we use 5 images from each video which all have the same label
        n_duplicates = 5
        labels=[i for i in labels for _ in np.arange(n_duplicates)] 
        #images
        all_image_names = [self.image_dir+img_name for img_name in listdir(os.getcwd()+'/'+self.image_dir)]
        self.image_names = all_image_names
        self.labels = labels 
        #shuffle all data, labels and images shuffled in same way by keeping same seed
        rng = np.random.randint(1000)
        np.random.seed(rng)
        np.random.shuffle(self.image_names)
        np.random.seed(rng)
        np.random.shuffle(self.labels)
        #train/test/validation set
        self.train_set = {'images':self.image_names[:7500], 'labels':self.labels[:7500]}
        self.val_set = {'images':self.image_names[7500:8750], 'labels':self.labels[7500:8750]}
        self.test_set = {'images':self.image_names[8750:], 'labels':self.labels[8750:]}


    def __double_generator(self, data):
        '''
        Returns batches of (images,labels)
        '''
        L = len(data['labels'])
        #this line is just to make the generator infinite, keras needs that    
        while True:
            batch_start = 0
            batch_end = self.batch_size
            while batch_start < L:
                limit = min(batch_end, L)
                images = np.array([mpimg.imread(img)[:,:,0] for img in data['images'][batch_start:limit]]).reshape(-1,160,160,1)
                labels = np.array(data['labels'][batch_start:limit])
                yield (images,labels)   
                batch_start += self.batch_size   
                batch_end += self.batch_size
    def get_generators(self):
        '''
        Returns generators for each train, validation and test set
        '''
        train_generator = self.__double_generator(self.train_set)
        val_generator = self.__double_generator(self.val_set)
        test_generator = self.__double_generator(self.test_set)
        return [train_generator, val_generator, test_generator]
    def get_set_length(self):
        '''
        Return train val and test set length
        '''
        return len(self.train_set['labels']),len(self.val_set['labels']),len(self.test_set['labels'])

class CreateKerasModel(object):
    def __init__(self):
        self.model = keras.Sequential()
        self.__setup()
    def __setup(self):
        #Model architecture
        #Conv1
        self.model.add(Conv2D(8, kernel_size=5, padding='SAME', activation='relu',input_shape=(160,160,1),name='conv1'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
        #Conv2
        self.model.add(Conv2D(16, kernel_size=5, padding='SAME', activation='relu') )
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
        #Conv3
        self.model.add(Conv2D(32, kernel_size=5, padding='SAME', activation='relu') )
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
        #Conv4
        self.model.add(Conv2D(32, kernel_size=5, padding='SAME', activation='relu') )
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'))
        #Flatten
        self.model.add(Flatten())
        #Dense1
        self.model.add(Dense(800, activation='relu'))
        #Dense2
        self.model.add(Dense(1,name="preds"))
        #optimizer and loss
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, train_generator, val_generator, test_generator, train_len, val_len, test_len):
        #define logs directory for tensorboard
        #tensorboard = TensorBoard(log_dir="logs/keras_runs")
        #define steps
        steps_per_epoch = math.ceil(train_len/FLAGS.batch_size)
        val_steps = math.ceil(val_len/FLAGS.batch_size)
        #Training
        self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=FLAGS.max_epochs, validation_data=val_generator, validation_steps=val_steps, verbose=1)#, callbacks=[tensorboard])
        #Evaluation
        test_steps = test_len/FLAGS.batch_size
        print(self.model.evaluate_generator(test_generator, steps=test_steps))
        print(self.model.metrics_names)
    

def main(argv):
    label_generator =  ImageLabelGenerator('cropSampled/', 'video_targets_minus1.csv', 'r')
    train_generator, val_generator, test_generator = label_generator.get_generators()
    keras_model = CreateKerasModel()
    train_len, val_len, test_len = label_generator.get_set_length()
    keras_model.train(train_generator, val_generator, test_generator, train_len, val_len, test_len)


if __name__ == '__main__':
    main(sys.argv[1:])