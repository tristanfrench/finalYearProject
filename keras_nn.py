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


FLAGS = tf.app.flags.FLAGS
#started at 13:15
# Optimisation hyperparameters batch size was 64 for nathan
tf.app.flags.DEFINE_integer('batch_size', 64, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_width', 160, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_height', 160, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_channels', 1, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num_classes', 1, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_integer('max_epochs', 50,'Number of mini-batches to train on. (default: %(default)d)')
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

    #print(model.predict_generator(test_generator, steps=test_steps))
    #print(test_data_labels)
    #print(np.mean(abs(np.transpose(np.array(model.predict_generator(test_generator, steps=test_steps)))-np.array(test_data_labels))))
    



    #plt.rcParams['figure.figsize'] = (18, 6)

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'conv1')

    # Swap softmax with linear
    #model.layers[layer_idx].activation = activations.linear
    #model = utils.apply_modifications(model)

    # This is the output node we want to maximize.
    #filter_idx = 0
    img = visualize_activation(model, layer_idx,filter_indices=[0])
    print(np.shape(img))
    plt.imshow(img[..., 0])
    plt.show()


    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'preds')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=val_x[idx])
    # Plot with 'jet' colormap to visualize as a heatmap.
    plt.imshow(grads, cmap='jet')


    # This corresponds to the Dense linear layer.
    for class_idx in np.arange(10): 
        indices = np.where(val_y[:, class_idx] == 1.)[0]
        idx = indices[0]

        f, ax = plt.subplots(1, 4)
        ax[0].imshow(val_x[idx][..., 0])
    
        for i, modifier in enumerate([None, 'guided', 'relu']):
            grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, 
            seed_input=val_x[idx], backprop_modifier=modifier)
            if modifier is None:
                modifier = 'vanilla'
            ax[i+1].set_title(modifier) 
            ax[i+1].imshow(grads, cmap='jet')


if __name__ == '__main__':
    tf.app.run(main=main)
