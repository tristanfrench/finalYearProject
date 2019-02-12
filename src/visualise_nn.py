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
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency
from collections import defaultdict


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
tf.app.flags.DEFINE_integer('log_frequency', 15,'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
'Directory where to write event logs and checkpoint. (default: %(default)s)')
run_log_dir = os.path.join(FLAGS.log_dir, 'ep_{ep}_bs_{bs}'.format(ep=FLAGS.max_epochs,bs=FLAGS.batch_size))



def readCsv(csv_file):
    data = pd.read_csv(csv_file)
    newData = data[['pose_1','pose_6']]
    return newData
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

def read_images(images):
    return mpimg.imread(images)[:,:,0]
def img_generator(filenames, labels, batch_size):
    L = len(filenames)

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = np.array([mpimg.imread(img)[:,:,0] for img in filenames[batch_start:limit]]).reshape(-1,160,160,1)
            Y = np.array(labels[batch_start:limit])

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size
def iter_occlusion(image, size=8):
    # taken from https://www.kaggle.com/blargl/simple-occlusion-and-saliency-maps

    occlusion = np.full((size * 5, size * 5, 1), [0.5], np.float32)
    occlusion_center = np.full((size, size, 1), [0.5], np.float32)
    occlusion_padding = size * 2

    # print('padding...')
    image_padded = np.pad(image, ( \
    (occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0) \
    ), 'constant', constant_values = 0.0)

    for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):

        for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
            tmp = image_padded.copy()

            tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
                x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
                = occlusion

            tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center

            yield x - occlusion_padding, y - occlusion_padding, \
                tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]
def main(_):
    tf.reset_default_graph()
    images,labels = image_preprocess()
    labels = labels[0]
    print(len(labels))
    rng = np.random.randint(1000)
    np.random.seed(rng)
    np.random.shuffle(images)
    np.random.seed(rng)
    np.random.shuffle(labels)
    train_data_images = images[:7500]
    train_data_labels = labels[:7500]
    val_data_images = images[7500:8750]
    val_data_labels = labels[7500:8750]
    #test_data_images = images[8750:]
    #test_data_labels = labels[8750:]
    test_data_images = images[9929:]
    test_data_labels = labels[9929:]  

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
    model.add(Dense(1,name="preds"))
    #optimizer and loss
    print('here')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    train_generator = img_generator(train_data_images,train_data_labels,FLAGS.batch_size)
    val_generator = img_generator(val_data_images,val_data_labels,FLAGS.batch_size)
    test_generator = img_generator(test_data_images,test_data_labels,FLAGS.batch_size)

    steps_per_epoch = math.ceil(len(train_data_images)/FLAGS.batch_size)
    val_steps = math.ceil(len(val_data_images)/FLAGS.batch_size)
    tensorboard = TensorBoard(log_dir="logs/keras_runs")
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=FLAGS.max_epochs, validation_data=val_generator, validation_steps=val_steps, verbose=1, callbacks=[tensorboard])
    test_steps = len(test_data_images)/FLAGS.batch_size
 
    print(model.evaluate_generator(test_generator, steps=test_steps))
    print(model.metrics_names)

    


    #video_2000_12_crop
    img_to_see = plt.imread("cropSampled/video_0099_8_crop.jpg")
    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'preds')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    #grads = visualize_saliency(model, layer_idx, filter_indices=None,seed_input=img_to_see[:,:,0].reshape([160,160,1]))

    f, ax = plt.subplots(1, 4)
    ax[0].imshow(img_to_see)

    for i, modifier in enumerate([None, 'guided', 'relu']):
        grads = visualize_saliency(model, layer_idx, filter_indices=0, 
        seed_input=img_to_see[:,:,0].reshape([160,160,1]), backprop_modifier=modifier)
        if modifier is None:
            modifier = 'vanilla'
        ax[i+1].set_title(modifier) 
        ax[i+1].imshow(grads, cmap='jet')
    plt.show()
    '''
    f, ax = plt.subplots(1, 4)
    ax[0].imshow(img_to_see)
    img_to_see = plt.imread("cropSampled/video_1447_7_crop.jpg")
    for i, modifier in enumerate([None, 'guided', 'relu']):
        grads = visualize_saliency(model, layer_idx, filter_indices=0, 
        seed_input=img_to_see[:,:,0].reshape([160,160,1]), backprop_modifier=modifier)
        if modifier is None:
            modifier = 'vanilla'
        ax[i+1].set_title(modifier) 
        ax[i+1].imshow(grads, cmap='jet')
    plt.show()
    '''

    i = 23 # for example
    data = plt.imread("cropSampled/video_0099_8_crop.jpg")
    print(model.predict(data[:,:,0].reshape([1,160,160,1])))
    label_99 = 2.544006547
    #correct_class = np.argmax(val_y[i])
    correct_class = 0
    # input tensor for model.predict
    #inp = data.reshape(1, 28, 28, 1)

    # image data for matplotlib's imshow
    #img = data.reshape(28, 28)

    # occlusion
    img_size = 160
    occlusion_size = 40

    print('occluding...')

    heatmap = np.zeros((img_size, img_size), np.float32)
    class_pixels = np.zeros((img_size, img_size), np.int16)

    '''
    counters = defaultdict(int)
    for x,y,img in iter_occlusion(data, size=occlusion_size):
        out = model.predict(img[:,:,0].reshape([1,160,160,1]))
        print(out)
        input("Press Enter to continue...")
    dfg
    '''
    for n, (x, y, img_float) in enumerate(iter_occlusion(data, size=occlusion_size)):

        X = img_float[:,:,0].reshape(1, 160,160, 1)
        out = model.predict(X)[[0]]
        #print('#{}: {} @ {} (correct class: {})'.format(n, np.argmax(out), np.amax(out), out[0][correct_class]))
        #print('x {} - {} | y {} - {}'.format(x, x + occlusion_size, y, y + occlusion_size))

        heatmap[y:y + occlusion_size, x:x + occlusion_size] = abs(out-label_99)
        #class_pixels[y:y + occlusion_size, x:x + occlusion_size] = np.argmax(out)
        #counters[np.argmax(out)] += 1
    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    tf.app.run(main=main)
