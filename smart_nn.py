import tensorflow as tf
import numpy as np
import pandas as pd
import os
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
#started at 13:15
# Optimisation hyperparameters batch size was 64 for nathan
tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_width', 160, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_height', 160, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_channels', 1, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num_classes', 1, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_integer('max_epochs', 30,'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 1,'Number of steps between logging results to the console and saving summaries (default: %(default)d)')



def readCsv(csv_file):
    data = pd.read_csv(csv_file)
    newData = data[['pose_1','pose_6']]
    return newData

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')


def deepnn(x_image):
    #x_image = tf.reshape(x_image, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
    # First convolutional layer - maps one image to 32 feature maps.
    with tf.variable_scope('Conv_1'):
        #Conv1
        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 8])
        b_conv1 = bias_variable([8])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv1)
        #POOL1 160*160*8
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')
        #Conv2 80*80*8
        W_conv2 = weight_variable([5, 5, 8, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv2)
        #Pool2 80*80*16
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')
        #Conv3 40*40*16
        W_conv3 = weight_variable([5, 5, 16, 32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv3)
        #Pool3 40*40*32
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')
        #Conv4 20*20*32
        W_conv4 = weight_variable([5, 5, 32, 32])
        b_conv4 = bias_variable([32])
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv4)
        #Pool4 20*20*32
        h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')

        #reshape 10*10*32
        h_final = tf.reshape(h_pool4, [-1,3200])   
        #FC1    
        w_1_dim = 800
        w_y_dim = 1
        w_1 = tf.Variable(tf.truncated_normal([3200, w_1_dim], stddev=0.1))
        b_1 = tf.Variable(tf.constant(0.1, shape=[w_1_dim]))
        h_fc1 = tf.nn.relu(tf.matmul(h_final, w_1) + b_1)
        #FC2
        w_y = tf.Variable(tf.truncated_normal([800,w_y_dim], stddev=0.1))
        b_y = tf.Variable(tf.constant(0.1, shape=[w_y_dim]))
        h_fcy = tf.matmul(h_fc1, w_y) + b_y
        return h_fcy

def image_preprocess():
    data_labels = readCsv("video_targets_minus1.csv")#_minus1.csv")
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
    allImages = ['cropSampled/'+f for f in listdir(os.getcwd()+'/'+'cropSampled'+'/') if isfile(join(os.getcwd()+'/'+'cropSampled'+'/', f))]
    new_df = np.transpose(new_df.as_matrix(columns=new_df.columns[:1]))
    return allImages,new_df
    
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image, label


def batch_this(filenames,labels,batch_size,n_files,repeat=1):
    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    dataset = dataset.shuffle(buffer_size=n_files) 
    dataset = dataset.map(parse_function)
    if repeat:
        dataset = dataset.batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    return iterator


def main(_):
    tf.reset_default_graph()
    images,labels = image_preprocess()
    labels = labels[0]
    print(labels[0])
    np.random.seed(0)
    np.random.shuffle(images)
    np.random.seed(0)
    np.random.shuffle(labels)
    train_data_images = images[:7500]
    train_data_labels = labels[:7500]
    val_data_images = images[7500:8750]
    val_data_labels = labels[7500:8750]
    test_data_images = images[8750:]
    test_data_labels = labels[8750:]

    train_img_pl = tf.placeholder(tf.string, [None])
    train_labels_pl= tf.placeholder(tf.float32, [None])
    val_img_pl = tf.placeholder(tf.string, [None])
    val_labels_pl= tf.placeholder(tf.float32, [None])
    test_img_pl = tf.placeholder(tf.string, [None])
    test_labels_pl= tf.placeholder(tf.float32, [None])
    #train set
    train_iterator = batch_this(train_img_pl,train_labels_pl,FLAGS.batch_size,len(train_data_images))
    train_batch = train_iterator.get_next()
    #validation set
    val_iterator = batch_this(val_img_pl,val_labels_pl,FLAGS.batch_size,len(val_data_images))
    val_batch = val_iterator.get_next()
    #test set
    img_number = len(test_data_images)
    test_iterator = batch_this(test_img_pl,test_labels_pl,FLAGS.batch_size,img_number,repeat=0)
    test_batch = test_iterator.get_next()

    with tf.variable_scope('inputs'):
        #x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width,FLAGS.img_height,FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    y_conv = deepnn(x)

    with tf.variable_scope('x_entropy'):
            cross_entropy = tf.losses.mean_squared_error(labels = y_,predictions = y_conv)

    
    # Define your AdamOptimiser, using FLAGS.learning_rate to minimixe the loss function
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    #global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step ,1000,0.8)
    #optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step)
    #calculate the prediction and the accuracy
    #correct_prediction = tf.placeholder(tf.float32, [1])
    #accuracy = tf.Variable(tf.float32, [1])
    #accuracy = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    accuracy = tf.losses.absolute_difference(y_conv,y_)
    #cnn_output = y_conv
    #accuracy = tf.reduce_mean(tf.cast(abs_diff, tf.float32))
    #loss_summary = tf.summary.scalar('Loss', cross_entropy)
    #acc_summary = tf.summary.scalar('Accuracy', accuracy)

    # summaries for TensorBoard visualisation
    #validation_summary = tf.summary.merge([img_summary, acc_summary])
    #training_summary = tf.summary.merge([img_summary, loss_summary])
    #test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    #saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge([loss_summary,acc_summary])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        for epoch in range(FLAGS.max_epochs):
            # Training
            sess.run(train_iterator.initializer,feed_dict={train_img_pl:train_data_images,train_labels_pl:train_data_labels})
            sess.run(val_iterator.initializer,feed_dict={val_img_pl:val_data_images,val_labels_pl:val_data_labels})
            while True:
                try:
                    [train_img,train_labels] = sess.run(train_batch)
                    train_labels = np.transpose(np.array([train_labels])) # makes it a column vector, required
                    _,train_summary = sess.run([optimizer,merged], feed_dict={x: train_img, y_: train_labels}) 
                except tf.errors.OutOfRangeError:
                    break
                #Validation Accuracy
                [val_images,val_labels] = sess.run(val_batch)
                val_labels = np.transpose(np.array([val_labels])) # makes it a column vector, required
                val_accuracy,_ = sess.run([accuracy,merged], feed_dict={x: val_images, y_: val_labels})
                if step % FLAGS.log_frequency == 0:
                    print(' step: %g,accuracy: %g' % (step,val_accuracy))
                step += 1
        ############################################################
        #                                                          #
        #                EVALUATION                                #
        #                                                          #
        ############################################################
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        sess.run(test_iterator.initializer,feed_dict={test_img_pl:test_data_images,test_labels_pl:test_data_labels})
        while evaluated_images != img_number:
            [test_images,test_labels] = sess.run(test_batch)
            test_labels = np.transpose(np.array([test_labels])) # makes it a column vector, required
            evaluated_images += len(test_labels)
            test_accuracy_temp = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})
            test_accuracy += test_accuracy_temp
            batch_count += 1
            
        
        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % (test_accuracy))

    print('done') 
        


if __name__ == '__main__':
    tf.app.run(main=main)

#RESULTS:
#ANGLES: err:4.8 it:1000 lr:1e-3 bs:128

#displacement:  err:1.5 it:1000 lr:1e-3 bs:128




