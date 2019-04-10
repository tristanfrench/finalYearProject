import tensorflow as tf
import numpy as np
import pandas as pd

import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras import activations
from keras.utils import to_categorical
#from keras.callbacks import TensorBoard
from sklearn.model_selection import KFold

import os
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from matplotlib.pyplot import imshow

import sys

#hyperparameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 64, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_integer('max_epochs', 100,'Number of mini-batches to train on. (default: %(default)d)')

class ImageLabelGenerator(object):
    def __init__(self, image_dir, label_dir, feature=0, categorize=1):
        #attributes
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.feature = feature
        self.image_names = None
        self.labels = None
        self.train_set = {}
        self.test_set = {}
        self.batch_size = FLAGS.batch_size
        self.categorize = categorize

        #setup
        if self.feature == 0:
            self.__data_setup_double()
        else:
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
        if self.categorize:
            self.labels = self.__categorize_labels(labels)
        else:
            self.labels = labels 
        #shuffle all data, labels and images shuffled in same way by keeping same seed
        self.__shuffle_data()

    def __data_setup_double(self):
        '''
        Defines image names and labels.
        Image names are strings with name of image folder such as: 'cropSampled/video_0001_10_crop.jpg'.
        Defines all data sets
        '''
        #labels
        imported_labels = pd.read_csv(self.label_dir)
        r = imported_labels['pose_1'].values
        theta = imported_labels['pose_6'].values
        labels = np.zeros([len(r),2])
        labels[:,0] = r
        labels[:,1] = theta
        #duplicate each label 5 times because we use 5 images from each video which all have the same label
        n_duplicates = 5
        labels=[i for i in labels for _ in np.arange(n_duplicates)] 
        #images
        all_image_names = [self.image_dir+img_name for img_name in listdir(os.getcwd()+'/'+self.image_dir)]
        self.image_names = all_image_names
        self.labels = labels 
        #shuffle all data, labels and images shuffled in same way by keeping same seed
        self.__shuffle_data()

    def __shuffle_data(self):
        rng = np.random.randint(1000)
        np.random.seed(rng)
        np.random.shuffle(self.image_names)
        np.random.seed(rng)
        np.random.shuffle(self.labels)
    def __categorize_labels(self, label):
        if self.feature == 'theta':
            for idx,i in enumerate(label):
                if i<-40:
                    label[idx] = 0
                elif i<-35:
                    label[idx] = 1
                elif i<-30:
                    label[idx] = 2
                elif i<-25:
                    label[idx] = 3
                elif i<-20:
                    label[idx] = 4
                elif i<-15:
                    label[idx] = 5
                elif i<-10:
                    label[idx] = 6
                elif i<-5:
                    label[idx] = 7
                elif i<0:
                    label[idx] = 8
                elif i<5:
                    label[idx] = 9
                elif i<10:
                    label[idx] = 10
                elif i<15:
                    label[idx] = 11
                elif i<20:
                    label[idx] = 12
                elif i<25:
                    label[idx] = 13
                elif i<30:
                    label[idx] = 14
                elif i<35:
                    label[idx] = 15
                elif i<40:
                    label[idx] = 16
                else:
                    label[idx] = 17
        else:
            for idx,i in enumerate(label):
                if i<-5:
                    label[idx] = 0
                elif i<-4:
                    label[idx] = 1
                elif i<-3:
                    label[idx] = 2
                elif i<-2:
                    label[idx] = 3
                elif i<-1:
                    label[idx] = 4
                elif i<0:
                    label[idx] = 5
                elif i<1:
                    label[idx] = 6
                elif i<2:
                    label[idx] = 7
                elif i<3:
                    label[idx] = 8
                elif i<4:
                    label[idx] = 9
                elif i<5:
                    label[idx] = 10
                elif i<6:
                    label[idx] = 11
                elif i<7:
                    label[idx] = 12
                elif i<8:
                    label[idx] = 13
                else:
                    label[idx] = 14
        return label

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
    
    def decode_hot(self, labels):
        if self.feature == 'classification':
            return -45+labels*5
        else:
            return -6+labels

    def get_generators(self, train_indices, test_indices):
        '''
        Returns generators for each train, validation and test set
        '''
        self.train_set = {'images':np.array(self.image_names)[train_indices],'labels':np.array(self.labels)[train_indices]}
        self.test_set = {'images':np.array(self.image_names)[test_indices],'labels':np.array(self.labels)[test_indices]}

        train_generator = self.__double_generator(self.train_set)
        test_generator = self.__double_generator(self.test_set)
        return [train_generator, test_generator]
    
    def get_set_length(self):
        '''
        Return train val and test set length
        '''
        return len(self.train_set['labels']),len(self.test_set['labels'])

class KerasModel(object):
    def __init__(self, model_type, output_type):
        self.model = None
        self.accuracy = []
        self.type = output_type
        '''
        if model_type == 'classification':
            self.__create_classifier()
        elif model_type == 'regression':
            self.__create_regressor()
        else:
            raise Exception(f'{model_type} is not a valid model name')
        '''

    def create_classifier(self):
        if self.type == 'double':
            output_shape = 2
        else:
            output_shape = 1
        #Model architecture
        self.model = keras.Sequential()
        #Conv1
        self.model.add(Conv2D(8, kernel_size=[5,20], padding='SAME', activation='relu',input_shape=(160,160,1),name='conv1'))
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
        self.model.add(Dense(output_shape, name="preds"))
        #optimizer and loss
        self.model.compile(optimizer='adam', loss='mse', metrics=['sparse_categorical_accuracy'])

    def create_regressor(self):
        if self.type == 'double':
            output_shape = 2
        else:
            output_shape = 1
        #Model architecture
        self.model = keras.Sequential()
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
        self.model.add(Dense(output_shape, name='preds'))
        #optimizer and loss
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    def __multi_accuracy(self,test_generator, test_len):
        results = np.zeros([test_len,2])
        my_it = 0
        for img, y in test_generator:
            for i in range(64):
                try:
                    out = self.model.predict(img[i].reshape(-1,160,160,1))[0]
                    results[my_it] = abs(out-y[i])
                    my_it += 1
                except Exception as e:
                    break
            if my_it == test_len:
                break
        #average error over test set
        return (np.mean(results[:,0]), np.mean(results[:,1]))

    def train(self, train_generator, test_generator, train_len, test_len):
        #define logs directory for tensorboard
        #tensorboard = TensorBoard(log_dir="logs/keras_runs")
        #define steps
        steps_per_epoch = math.ceil(train_len/FLAGS.batch_size)
        #Training
        self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=FLAGS.max_epochs, verbose=2)#, callbacks=[tensorboard])
        #Evaluation
        test_steps = test_len/FLAGS.batch_size
        if self.type == 'single':
            loss, acc = self.model.evaluate_generator(test_generator, steps=test_steps)
            self.accuracy.append(acc)
            print(loss, acc)
            print(self.model.metrics_names)
        else:
            r_error, theta_error = self.__multi_accuracy(test_generator, test_len)
            print(r_error,theta_error)
            self.accuracy.append([r_error,theta_error])
    
    def kill_model(self):
        self.model = None
    

def main(argv):
    label_generator =  ImageLabelGenerator('cropSampled/', 'video_targets_minus1.csv', 'theta', categorize=0)
    model_type = 'regression'
    output_type = 'single'
    keras_model = KerasModel(model_type,output_type)
    kf = KFold(n_splits=10, shuffle=True)
    for idx, (train_indices, test_indices) in enumerate(kf.split(label_generator.image_names, label_generator.labels)):
        print('Training on fold'+str(idx+1)+'/10...')
        # Generate batches from indices
        train_generator, test_generator = label_generator.get_generators(train_indices, test_indices)
        #Create model
        keras_model.create_regressor()
        train_len, test_len = label_generator.get_set_length()
        keras_model.train(train_generator, test_generator, train_len, test_len)
        keras_model.kill_model()
    
    print(keras_model.accuracy)
    print('theta 100ep mean=',np.mean(keras_model.accuracy,0))
    '''
    #check a prediction:
    img_to_see = plt.imread('cropSampled/video_1794_8_crop.jpg')[:][:,:,0]
    X = img_to_see.reshape(1, 160,160, 1)
    out = keras_model.model.predict(X)
    print(label_generator.decode_hot(out)) 
    img_to_see = plt.imread('cropSampled/video_0794_9_crop.jpg')[:][:,:,0]
    X = img_to_see.reshape(1, 160,160, 1)
    out = keras_model.model.predict(X)
    #keras_model.model.save(f'trained_models/categ.h5')
    print(label_generator.decode_hot(out)) 
    '''
    



if __name__ == '__main__':
    main(sys.argv[1:])


#results:
#10ep r [0.23, 0.32, 0.50, 0.26, 0.27, 0.36, 0.36, 0.24, 0.28, 0.36]
#50ep r [0.74,0.28,0.45,0.24,0.16,0.19,0.21,0.13,0.13,0.16] mean 0.27 median 0.2
#100 ep r [0.21, 0.16, 0.14, 0.33, 0.17, 0.21, 0.27, 0.16, 0.25, 0.14]

#10ep theta [1.80, 2.26, 2.52, 1.31, 1.40, 1.61, 1.73, 1.63, 1.77, 1.49]
#50ep theta[0.82, 22.37, 1.59, 0.912, 22.57, 22.85, 1.02, 1.68, 1.05, 1.11]
#100 ep theta [0.74, 0.99, 22.91, 0.90, 0.90, 1.023, 0.88, 1.47, 1.03, 0.92]
#weird result: theta
# [14.199, 2.9977, 2.3615, 2.0212, 1.8409, 1.6869, 1.5961, 1.5314, 1.5186, 1.4498, 1.3962, 1.5541, 1.4422, 1.3437, 1.3566, 1.3889, 1.365, 1.2986, 1.2142, 1.1465, 1.065, 1.0597, 1.0214, 1.0102, 0.9744, 0.9719, 0.9659, 0.9788, 0.9693, 1.0334, 2.9091, 22.4753, 22.4766, 22.4772, 22.4775, 22.4776, 22.4777, 22.4777, 22.4777, 22.4777, 22.4778, 22.4778, 22.4778, 22.4778, 22.4778, 22.4778, 22.4777, 22.4777, 22.4777, 22.4777] 

#double 10 ep
#[[0.40615642910558303, 1.651251674039564], [0.35700454032079515, 1.4364349266894385], [0.4611907497247906, 1.9815532311021176], [0.29057074755010026, 1.3277033459302134], [0.3467601213353207, 1.7254202273478323], [0.3655495394668749, 1.476608993864042],
#[0.4533175195935343, 2.8588751086283732], [0.4388708920837557, 2.2388375414827415], [0.472477407390929, 2.2318424523486198], [0.312624640946713, 1.643917854311274]]
#theta 100ep mean= [0.39045226 1.85724454]

#double 50 ep
'''
[[0.2270851126815107, 1.0754155113247499], [0.3597043034642528, 1.0536525467939772], [0.3437720492597648, 1.2030333714272559],
[0.2102534972408078, 1.1327671315221992], [0.28693500201065636, 1.4567695883085134], [0.3010229023080997, 1.0837306313025352],
[0.2718577189204042, 1.1293397151227962], [0.21242246289314348, 0.9679065260811791], [0.24387909756775877, 1.05468332909014], [3.907444548038397, 22.651023749419785]]
'''

#double 100 ep
'''
[[0.17834223084933046, 0.7246390885436051], [0.23002465958828686, 0.8171563322895091], [0.22531053139966092, 1.0197750759831827], [3.6958066190761514, 22.28227279116719], [3.74518570368829, 22.65783426675385], [3.9070902985882516, 22.675373265289064], 
[3.9195956791632756, 22.16931346264702], [0.20382954094178368, 0.8813804324853676], [0.1939251668346117, 0.856512263433736], [0.2455364326345564, 0.9663450797724967]]
'''

#10 ep kernel r
#median 0.29
#50 ep kernel r
#median 0.18
#100 ep kernel r
#median 0.19


#10 ep kernel theta
#median 1.72
#50 ep kernel theta
#median 1.07
#100 ep kernel theta 
#[1.0645147771835328, 22.253843276977538, 22.364766357421875, 0.7775294423103333, 22.33252522277832, 22.252103020836998, 1.5656768772098515, 0.8450547164386218, 22.146669962504006, 1.0456013768404215]