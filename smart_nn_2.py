import tensorflow as tf
import numpy as np
import pandas as pd
import os
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import math

FLAGS = tf.app.flags.FLAGS
#started at 13:15
# Optimisation hyperparameters batch size was 64 for nathan
tf.app.flags.DEFINE_integer('batch_size', 64, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_width', 160, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_height', 160, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_channels', 1, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num_classes', 1, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_integer('max_epochs', 5,'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_on', 1,'Save logs or not. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 117,'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
'Directory where to write event logs and checkpoint. (default: %(default)s)')
run_log_dir = os.path.join(FLAGS.log_dir, 'ep_{ep}_bs_{bs}'.format(ep=FLAGS.max_epochs,bs=FLAGS.batch_size))


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


def deepnn(x,visualise=0):
    #x_image = tf.reshape(x_image, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
    # First convolutional layer - maps one image to 32 feature maps.
    #Conv1
    conv1 = tf.layers.conv2d(
    inputs=x,
    filters=8,
    kernel_size=[5,5],
    padding='same',
    use_bias=False,
    name='conv1'
    )
    #if visualise:
    #    return conv1
    conv1 = tf.nn.relu(conv1)
    #POOL1 160*160*8
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=[2,2],
        name='pool1'
    )
    #Conv2 80*80*8
    conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=16,
    kernel_size=[5,5],
    padding='same',
    use_bias=False,
    name='conv2'
    )
    conv2 = tf.nn.relu(conv2) 
    #Pool2 80*80*16
    pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=[2,2],
    name='pool2'
    )
    #Conv3 40*40*16
    conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=32,
    kernel_size=[5,5],
    padding='same',
    use_bias=False,
    name='conv3'
    )
    conv3 = tf.nn.relu(conv3) 
    #Pool3 40*40*32
    pool3 = tf.layers.max_pooling2d(
    inputs=conv3,
    pool_size=[2,2],
    strides=[2,2],
    name='pool3'
    )
    #Conv4 20*20*32
    conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=32,
    kernel_size=[5,5],
    padding='same',
    use_bias=False,
    name='conv4'
    )
    conv4 = tf.nn.relu(conv4) 
    #Pool4 20*20*32
    pool4 = tf.layers.max_pooling2d(
    inputs=conv4,
    pool_size=[2,2],
    strides=[2,2],
    name='pool4'
    )
    #Reshape 10*10*32
    h_final = tf.reshape(pool4, [-1,3200])    
    #Fully connected
    fc1 = tf.layers.dense(h_final,units=800,activation=tf.nn.relu)
    fcy = tf.layers.dense(fc1,units=1) 
    return fcy,conv1,conv4

def image_preprocess():
    data_labels = readCsv("video_targets_minus1.csv")
    df = data_labels[['pose_1','pose_6']]
    df.columns =['r','theta']

    a=[]
    n_duplicates = 5
    for index, row in df.iterrows():
        for i in range(0,n_duplicates):
            a.append([row['r'],row['theta']])       
    new_df = pd.DataFrame(a,columns=['r','theta'])
    allImages = ['cropSampled/'+f for f in listdir(os.getcwd()+'/'+'cropSampled'+'/') if isfile(join(os.getcwd()+'/'+'cropSampled'+'/', f))]
    new_df = np.transpose(new_df.as_matrix(columns=new_df.columns[:1]))#r
    #new_df = np.transpose(new_df.as_matrix(columns=new_df.columns[1:]))#theta
    return allImages,new_df
    
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image, label


def batch_this(filenames,labels,batch_size,n_files,repeat=0):
    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    dataset = dataset.shuffle(buffer_size=n_files) 
    dataset = dataset.map(parse_function)
    if repeat:
        dataset = dataset.batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    return iterator




def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")



def main(_):
    tf.reset_default_graph()
    images,labels = image_preprocess()
    labels = labels[0]
    #min max rescaling:
    #labels = (labels-min(labels))/(max(labels)-min(labels))
    print(labels)
    rng = np.random.randint(1000)
    np.random.seed(rng)
    np.random.shuffle(images)
    np.random.seed(rng)
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
    val_iterator = batch_this(val_img_pl,val_labels_pl,FLAGS.batch_size,len(val_data_images),repeat=1)
    val_batch = val_iterator.get_next()
    #test set
    img_number = len(test_data_images)
    test_iterator = batch_this(test_img_pl,test_labels_pl,FLAGS.batch_size,img_number)
    test_batch = test_iterator.get_next()

    log_per_epoch = int(1000/FLAGS.max_epochs)
    log_frequency = round(len(train_data_labels)/FLAGS.batch_size/log_per_epoch)
    if log_frequency == 0 :
        log_frequency += 1

    with tf.variable_scope('inputs'):
        #x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width,FLAGS.img_height,FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    y_conv,lol,lol1 = deepnn(x)

    with tf.variable_scope('x_entropy'):
            cross_entropy = tf.losses.mean_squared_error(labels = y_,predictions = y_conv)

    
    # Define your AdamOptimiser, using FLAGS.learning_rate to minimixe the loss function
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    accuracy = tf.losses.absolute_difference(y_conv,y_)
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge([loss_summary,acc_summary])
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph,flush_secs=5)
        val_writer = tf.summary.FileWriter(run_log_dir + '_val', sess.graph,flush_secs=5)
        sess.run(tf.global_variables_initializer())
        step = 0
        for epoch in range(FLAGS.max_epochs):
            # Training
            sess.run(train_iterator.initializer,feed_dict={train_img_pl:train_data_images,train_labels_pl:train_data_labels})
            sess.run(val_iterator.initializer,feed_dict={val_img_pl:val_data_images,val_labels_pl:val_data_labels})
            print('Epoch:',epoch)
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
                val_accuracy,val_summary = sess.run([accuracy,merged], feed_dict={x: val_images, y_: val_labels})
                if step % FLAGS.log_frequency == 0:
                    print(' step: %g,accuracy: %g' % (step,val_accuracy))
                if step % log_frequency==0 and FLAGS.log_on==1:
                    train_writer.add_summary(train_summary,step)
                    val_writer.add_summary(val_summary,step)
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
            
        test_images[0]
        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % (test_accuracy))
        units = sess.run(lol1,feed_dict={x:np.reshape(test_images[0],[1,160,160,1])})
        plotNNFilter(units)
        plt.show()

        #plotNNFilter(units)
        #print(weights)
        #with tf.variable_scope('conv1',reuse=True) as scope:
            #tf.get_variable_scope().reuse_variables()
            #weights = tf.get_variable('kernel:0')
            #print(weights)
        #print(tf.get_variable('conv1'))



    print('done') 
        


if __name__ == '__main__':
    tf.app.run(main=main)

#RESULTS:
#displacement 5 epoch: 0.58 0.491 0.739 0.409 0.515 0.447 0.359 0.390 0.351 .542  mean:0.48
#100epoch: displacement: 0.166 0.195
#angle 5 epoch: 2.351 2.543 3.663 2.103 2.766 3.205 1.758 2.030 1.969 1.700 mean:2.6



