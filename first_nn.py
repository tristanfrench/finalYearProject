import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

FLAGS = tf.app.flags.FLAGS

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 4, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-4, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 640, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 480, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 1, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 1, 'Number of classes (default: %(default)d)')

    
def parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  #image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_decoded, label

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



def deepnn(x):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.
  Args:
      x: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
        number of pixels in a standard CIFAR10 image.
  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the object images into one of 10 classes
        (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
      img_summary: a string tensor containing sampled input images.
    """
    # Reshape to use within a convolutional neural net.  Last dimension is for
    # 'features' - it would be 1 one for a grayscale image, 3 for an RGB image,
    # 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
    #img_summary = tf.summary.image('Input_images', x_image)
    with tf.variable_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv1)

        # Pooling layer - downsamples by 2X.
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
        w_y_dim = 10

        w_1 = tf.Variable(tf.truncated_normal([4096, w_1_dim], stddev=0.1))
        b_1 = tf.Variable(tf.constant(0.1, shape=[w_1_dim]))
        h_fc1 = tf.nn.relu(tf.matmul(h_final, w_1) + b_1)
        w_y = tf.Variable(tf.truncated_normal([1024,w_y_dim], stddev=0.1))
        b_y = tf.Variable(tf.constant(0.1, shape=[w_y_dim]))
        h_fcy = tf.matmul(h_fc1, w_y) + b_y 
        return h_fcy

def main(_):
        sess = tf.Session()
        data_labels = readCsv("video_targets_minus1.csv")
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
        #print(new_df['r'][0])
        mainDir = 'collectCircleTapRand_08161204'
        imageDir = mainDir+'/extractedImages/'
        myPath = os.getcwd()+'/'+imageDir
        allImages = [f for f in listdir(myPath) if isfile(join(myPath, f))]
        dataset = tf.data.Dataset.from_tensor_slices((allImages, new_df['r']))
        dataset = dataset.map(parse_function)
        #x:_ondisk_parse_(x)).shuffle(True).batch(batch_size)
        batched_dataset = dataset.batch(4)

        iterator = batched_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        print(next_element)
        print(next_element)



        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                #sess.run(lol)
        print('done')

def main(_):
    tf.reset_default_graph()

    # Import data
    #cifar = cf.cifar10(batchSize=FLAGS.batch_size, downloadDir=FLAGS.data_dir)
    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    # Build the graph for the deep net
    y_conv, img_summary = deepnn(x)

    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        #change to mean squared
    
    # Define your AdamOptimiser, using FLAGS.learning_rate to minimixe the loss function
    #optimiser = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step ,1000,0.8)
    optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step)

    #calculate the prediction and the accuracy
    #correct_prediction = tf.placeholder(tf.float32, [1])
    #accuracy = tf.Variable(tf.float32, [1])
    accuracy = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir +'_validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using train set
            (trainImages, trainLabels) = cifar.getTrainBatch()
            (testImages, testLabels) = cifar.getTestBatch()
            
            _, summary_str = sess.run([optimiser, training_summary], feed_dict={x: trainImages, y_: trainLabels})
            
           
            if step % (FLAGS.log_frequency + 1)== 0:
                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                validation_accuracy, summary_str = sess.run([ accuracy,validation_summary], feed_dict={x: testImages, y_: testLabels})

                print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy),sess.run(learning_rate))
                summary_writer_validation.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Testing

        # resetting the internal batch indexes
        cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        # don't loop back when we reach the end of the test set
        while evaluated_images != cifar.nTestSamples:
            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            evaluated_images = evaluated_images + testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)

if __name__ == '__main__':
    tf.app.run(main=main)





