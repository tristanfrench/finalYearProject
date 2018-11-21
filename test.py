import tensorflow as tf
import numpy as np
import pandas as pd
import os
import os.path
from os import listdir
from os.path import isfile, join

def image_preprocess():
    data_labels = readCsv("video_targets.csv")#_minus1.csv")
    df = data_labels[['pose_1','pose_6']]
    df.columns =['r','theta']

    a=[]
    n_duplicates = 5
    for index, row in df.iterrows():
        for i in range(0,n_duplicates):
            a.append([row['r'],row['theta']])       
    new_df = pd.DataFrame(a,columns=['r','theta'])
    np.random.seed(0)
    #data = data.sample(frac=1).reset_index(drop=True)#shuffles data but keeps indices in place
    mainDir = 'collectCircleTapRand_08161204'
    imageDir = mainDir+'/extractedImages/'
    #myPath = os.getcwd()+'/'+imageDir
    allImages = [f for f in listdir(os.getcwd()+'/'+'imageSample'+'/') if isfile(join(os.getcwd()+'/'+'imageSample'+'/', f))]
    return allImages,new_df

def readCsv(csv_file):
    data = pd.read_csv(csv_file)
    newData = data[['pose_1','pose_6']]
    return newData
'''
def parse_function(filename):
  #image_string = tf.read_file(filename)
  #print('LOOOOOL1') 
  image_decoded = tf.image.decode_jpeg(filename,1)
  #print('LOOOOOL2')
  #img = tf.cast(image_decoded, tf.float32)
  #image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_decoded
'''
def get_images(allImages,batch_size):
    #dataset = tf.contrib.data.Dataset.from_tensor_slices((allImages, allLabels['r'][0:5]))
    #dataset = tf.contrib.data.Dataset.from_tensor_slices(allImages)
    #dataset = dataset.map(parse_function)
    dataset = parse_function(allImages)
    #batched_dataset = dataset.batch(batch_size)
    #iterator = batched_dataset.make_one_shot_iterator()
    return dataset#iterator.get_next()

#images,_ = image_preprocess()
# step 1
filenames = tf.constant(['img_01.jpg', 'img_02.jpg', 'img_03.jpg', 'img_04.jpg'])
labels = tf.constant([0, 1, 0, 1])

# step 2: create a dataset returning slices of `filenames`
dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))

# step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image, label

dataset = dataset.map(_parse_function)
dataset = dataset.batch(2)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

with tf.Session() as sess:
    print(sess.run([images, labels]))
    