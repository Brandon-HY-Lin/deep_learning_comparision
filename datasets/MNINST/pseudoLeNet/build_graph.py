#   The symbols are adapted from CS231n 
#       (http://cs231n.github.io/convolutional-networks/)
#
#
#   build CNN with 2 conv layers and 2 fully connected layers
#       input: Height x Width x Color= 28x28x1
#       C1: 
#           conv1: F=5, S=1 (field size=5, stride=1), padding=2 (SAME)
#               output: 28x28x6
#               params=5x5x1x6 + 6= 156
#       S2:
#           pool: F=2, S=2, padding=0 
#               output: 14x14x6
#       C3:
#           conv2: F=5, S=1, padding=0 (VALID)
#               output: 10x10x16
#               params = 5*5*6*16+16= 2,416
#       S4:
#           pool: F=2, S=2, padding=0
#               output: 5x5x16
#       C5:
#           conv3: F=5, S=1, padding=0 (VALID)
#               output = 1x1x120 
#               params = 5x5x16x120+ 120= 48,120
#       F6:
#           FC1: 120x84
#               params= 120x84 + 84= 10,164
#       Output:
#           84x10
#
#       Total params = 156 + 2,416 + 48,120 + 10,164 = 60,856
#

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import mnist

# Allocate tf.Variable for weight adn bias
# Return W, b
def get_variables(shape_weight, dtype=tf.float32):
    W_init = tf.truncated_normal(shape_weight, stddev=0.1)
    W = tf.Variable(W_init, dtype)

    b_init = tf.constant(0.1, shape=[shape_weight[-1]])
    b = tf.Variable(b_init, dtype)

    return W, b

def conv2d(x, W, b, activation, padding='SAME'):
    x_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    return activation(x_conv + b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, 
                        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                        padding='SAME')

def build_graph(is_learning, enable_bn):
    collection_train = 'train'
    collection_all = 'all'
    collection_layers = 'layers'

    dtype = tf.float32
    #activation = tf.nn.tanh
    activation = tf.nn.relu

    x = tf.placeholder(dtype, [None, mnist.IMAGE_PIXELS])
    y_ = tf.placeholder(dtype, [None, mnist.NUM_CLASSES])

    x_image = tf.reshape(x, [-1, mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1])

    # C1
    with tf.name_scope('C1'):
        W1, b1 = get_variables([5, 5, 1, 6])
        h1 = conv2d(x_image, W1, b1, activation)
        x_conv = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')

        tf.summary.image('W1', tf.reshape(W1, [-1, 5, 5, 1]), 
                        max_outputs=6, collections=[collection_train])
        #tf.summary.histogram('b1', b1)

        tf.summary.image('input', tf.reshape(x_image, [-1, 28, 28, 1]), 
                            collections=[collection_layers])

        tf.summary.image('h1', tf.transpose(h1, [3, 1, 2, 0]), 
                            collections=[collection_layers])
        tf.summary.image('x_conv', tf.transpose(x_conv, perm=[3, 1, 2, 0]), 
                            max_outputs=6, collections=[collection_layers])

    # S2
    with tf.name_scope('S2'):
        h2 = max_pool_2x2(h1)
        tf.summary.image('h2', tf.transpose(h2, [3, 1, 2, 0]), 
                            collections=[collection_layers])

    # C3
    with tf.name_scope('C3'):
        W3, b3 = get_variables([5, 5, 6, 16])
        h3 = conv2d(h2, W3, b3, activation, padding='VALID')

        tf.summary.image('W3', tf.reshape(W3, [-1, 5, 5, 1]), 
                        max_outputs=97, collections=[collection_train])

        tf.summary.image('h3', tf.transpose(h3, [3, 1, 2, 0]), 
                            collections=[collection_layers])

    # S4
    with tf.name_scope('S4'):
        h4 = max_pool_2x2(h3)
        tf.summary.image('h4', tf.transpose(h4, [3, 1, 2, 0]), 
                            collections=[collection_layers])

    # C5
    with tf.name_scope('C5'):
        W5, b5 = get_variables([5, 5, 16, 120])
        h5 = conv2d(h4, W5, b5, activation, padding='VALID')

        tf.summary.image('W5', tf.reshape(W5, [-1, 5, 5, 1]),
                        collections=[collection_train])

        #tf.summary.histogram('b5', b5)

        h_flatten = tf.reshape(h5, [-1, 120])

    # F6
    with tf.name_scope('F6'):
        W6, b6 = get_variables([120, 84])
        h6 = tf.matmul(h_flatten, W6) + b6

        #tf.summary.histogram('W6', W6)
        #tf.summary.histogram('b6', b6)

    # F7
    with tf.name_scope('F7'):
        W7, b7 = get_variables([84, 10])
        h7 = tf.matmul(h6, W7) + b7
        y_out = h7

        #tf.summary.histogram('W7', W7)
        #tf.summary.histogram('b7', b7)


    with tf.name_scope('loss'):
        loss = tf.reduce_mean( 
                    tf.nn.softmax_cross_entropy_with_logits(
                                labels=y_, logits=y_out)
                    )

        tf.summary.scalar('loss', loss, 
                            collections=[collection_all])

    with tf.name_scope('predict'):
        predict = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(predict, dtype))

        tf.summary.scalar('accuracy', accuracy, 
                            collections=[collection_all])

    return (x, y_), loss, accuracy, tf.train.Saver()
