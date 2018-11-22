import tensorflow as tf
import numpy as np
import pandas as pd
import os
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.image as mpimg




class Dataset:
    def __init__(self,data,batch_size):
        self.batch_count = 0
        self.data = data
        #self.row, self.col, self.depth = np.shape(data)
        self.batch_size = batch_size

    def get_next_batch(self):
        #if self.batch_count >
        batch = self.data[self.batch_count:self.batch_count + self.batch_size]
        self.batch_count += self.batch_size
        return batch


def readCsv(csv_file):
    data = pd.read_csv(csv_file)
    newData = data[['pose_1','pose_6']]
    return newData

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 2, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-3, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width',32 ,'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 1, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')



run_log_dir = os.path.join(FLAGS.log_dir, 'exp_BN_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate))

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')

def deepnn(x_image):
    #x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
    #print('deepnn')
    #print(np.shape(x_image))

    # First convolutional layer - maps one image to 32 feature maps.

    with tf.variable_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')

        # You need to continue building your convolutional network!
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv2)
        # Pooling layer - downsamples by 2X.
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')
        h_final = tf.reshape(h_pool2, [-1,4096])       
        w_1_dim = 1024
        w_y_dim = 1
        w_1 = tf.Variable(tf.truncated_normal([4096, w_1_dim], stddev=0.1))
        b_1 = tf.Variable(tf.constant(0.1, shape=[w_1_dim]))
        h_fc1 = tf.nn.relu(tf.matmul(h_final, w_1) + b_1)
        w_y = tf.Variable(tf.truncated_normal([1024,w_y_dim], stddev=0.1))
        b_y = tf.Variable(tf.constant(0.1, shape=[w_y_dim]))
        h_fcy = tf.matmul(h_fc1, w_y) + b_y
        return h_fcy

###############
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
    allImages = [mpimg.imread('imageSample/'+f) for f in listdir(os.getcwd()+'/'+'imageSample'+'/') if isfile(join(os.getcwd()+'/'+'imageSample'+'/', f))]
    new_df = new_df.as_matrix(columns=new_df.columns[1:])
    #print(np.shape(new_df))
    return allImages,new_df


def main(_):
    tf.reset_default_graph()

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width,FLAGS.img_height,FLAGS.img_channels])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    # Build the graph for the deep net
    y_conv = deepnn(x)
    with tf.variable_scope('x_entropy'):
            cross_entropy = tf.losses.mean_squared_error(labels = y_,predictions = y_conv)
        
    
    optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    images,labels = image_preprocess()
    print(np.shape(images))
    #print(np.shape(labels))
    imgsize= 32
    a = np.arange(imgsize*imgsize*3).reshape([imgsize,imgsize,3])
    images = Dataset([a,a,a,a],2)
    #images = Dataset(images,FLAGS.batch_size)
    labels = Dataset(labels,FLAGS.batch_size)
    #aa=images.get_next_batch()
    #print(np.shape(aa))
    
    #print(np.shape(labels.get_next_batch()))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([optimiser], feed_dict={x: images.get_next_batch(), y_: labels.get_next_batch()})
    
           



if __name__ == '__main__':
    tf.app.run(main=main)
