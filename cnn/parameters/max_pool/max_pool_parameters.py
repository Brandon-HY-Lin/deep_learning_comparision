#Copyright 2018 Brandon H. Lin. {brandon.hy.lin.0@gamil.com}
#
#Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
#-------------------------------------------------------------------------
#
# This work is inspired by the following guideline in CS231n in Stanford University
#	
# "It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with F=3,S=2 (also called overlapping pooling), and more commonly F=2,S=2. Pooling sizes with larger receptive fields are too destructive."
#
#	(http://cs231n.github.io/convolutional-networks/)
#
###########################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data, mnist
from collections import defaultdict

def get_images():
    PATH='./MNIST_data'
    num_images = 10
    data_set = input_data.read_data_sets(PATH)
    line_images = data_set.train.images[0:num_images]
    images = line_images.reshape( 
                        [-1, mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1])
    return images

def build_graph(settings, img_width, img_height, color):
    # create max_pool for different settings
    dtype = tf.float32

    x = tf.placeholder(dtype=dtype, 
                            shape=[None, img_width, img_height, color])

    y_dict = defaultdict()
    for title, parameters in settings.items():
        y = x

        for p in parameters:
            ksize = p['ksize']
            strides = p['strides']
            padding = p['padding']
            y = tf.nn.max_pool(y, 
                               ksize=ksize, strides=strides, 
                               padding=padding)

        name = 'y_' + title
        tf.summary.image(name, y)

        y_dict[title] = y

    return x, y_dict

def max_pool_images(x, y_dict, image_data):
    # reset default graph
    # start a session
    # init global variables
    # iterate tensors
    #   evaluate tensors
    #   store images to dict


    #tf.reset_default_graph()
    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./tmp/log/')
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        images_output = defaultdict()
        index = 0
        for title, y in y_dict.items():
            y_val = sess.run(fetches=[y], 
                                feed_dict={x: image_data})

            s = sess.run(fetches=merged_summary,
                        feed_dict={x: image_data})

            writer.add_summary(s, index)
            index += 1

            images_output[title] = y_val[0][0][:,:,0] # get 1st image
        
    return images_output

def plot_image_sets(image_sets, n_cols, width, height):
    num_sets = len(image_sets) 
    n_rows = (num_sets + n_cols - 1)  / n_cols

    index = 1

    plt.figure(1)
   
    for title, image in sorted(image_sets.items()):
        ax = plt.subplot(n_rows, n_cols, index)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        plt.title(title, fontsize=12)
        #plt.imshow(image, cmap='gray')
        plt.imshow(image)
        index += 1

    plt.tight_layout()
    plt.show()
        

def main():
    # get image data
    # define settings of max_pool
    # build graph
    #   iterate different settings of max_pool (ksize, strides, padding)
    # evaluate  graph
    # render plot

    PATH_MODEL='./tmp/model_'

    images = get_images()

    print type(images)
    print images.shape

    settings_ = defaultdict(list)
    settings_['1-1: F3_S2_SAME'].append(
                                    {'ksize':[1, 3, 3, 1], 
                                    'strides':[1, 2, 2, 1],
                                    'padding':'SAME'})

    settings_['1-2: F3_S2_SAME x2'].append(
                                    {'ksize':[1, 3, 3, 1], 
                                    'strides':[1, 2, 2, 1],
                                    'padding':'SAME'})
    settings_['1-2: F3_S2_SAME x2'].append(
                                    {'ksize':[1, 3, 3, 1], 
                                    'strides':[1, 2, 2, 1],
                                    'padding':'SAME'})

    settings_['2-1: F2_S2_SAME'].append({'ksize':[1, 2, 2, 1], 
                                    'strides':[1, 2, 2, 1],
                                    'padding':'SAME'})

    settings_['2-2: F2_S2_SAME x2'].append({'ksize':[1, 2, 2, 1], 
                                    'strides':[1, 2, 2, 1],
                                    'padding':'SAME'})

    settings_['2-2: F2_S2_SAME x2'].append({'ksize':[1, 2, 2, 1], 
                                    'strides':[1, 2, 2, 1],
                                    'padding':'SAME'})

    settings_['3-1: LeNet5_F5_S1_VALID'].append(
                                    {'ksize':[1, 5, 5, 1], 
                                    'strides':[1, 1, 1, 1],
                                    'padding':'VALID'})

    settings_['3-2: LeNet5_F5_S1_VALID x2'].append(
                                    {'ksize':[1, 5, 5, 1], 
                                    'strides':[1, 1, 1, 1],
                                    'padding':'VALID'})
    settings_['3-2: LeNet5_F5_S1_VALID x2'].append(
                                    {'ksize':[1, 5, 5, 1], 
                                    'strides':[1, 1, 1, 1],
                                    'padding':'VALID'})


    img_width = mnist.IMAGE_SIZE
    img_height = mnist.IMAGE_SIZE
    color = 1

    
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        x, y_dict  = build_graph(settings_, img_width, img_height, color)
        image_sets = max_pool_images(x, y_dict, images)

    image_sets['1-0: origin'] = images[0][:,:,0]
    image_sets['2-0: origin'] = images[0][:,:,0]
    image_sets['3-0: origin'] = images[0][:,:,0]
    n_cols = 3
    plot_image_sets(image_sets, n_cols, img_width, img_height)

if __name__ == '__main__':
    main()
