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
import numpy as tf

# Allocate tf.Variable for weight adn bias
# Return W, b
def conv_variables(shape_weight, dtype=tf.float32):
    W_init = tf.truncated_normal(shape_weight, stddev=0.1)
    W = tf.Variable(W_init, dtype)

    b_init = tf.constant(0.1, [shape_weight[-1]])
    b = tf.Variable(b_init, dtype)

    return W, b

def conv2d(x, W, b, activation, padding='SAME'):
    x_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    return activation(tf.matmul(x_conv, W) + b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, 
                        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                        padding='SAME')

def build_graph(is_learning, enable_bn, SIZES):
    dtype = tf.float32
    activation = tf.nn.tanh

    x = tf.placeholder(dtype, [None, mnist.IMAGE_PIXELS])
    y_ = tf.placeholder(dtype, [None, mnist.NUM_CLASSES])

    x_image = tf.reshape(x, [-1, mnist.IMAGE_SIZE< mnist.IMAGE_SIZE, 1])

    # C1
    with tf.name_scope('C1'):
        W1, b1 = conv_variables([5, 5, 1, 6])
        h1 = conv2d(x_image, W1, b1, activation)

        tf.summary.image('W1', W1)
        tf.summary.histogram('b1', b1)

    # S2
    with tf.name_scope('S2'):
        h2 = max_pool_2x2(h1)

    # C3
    with tf.name_scope('C3'):
        W3, b3 = conv_variables([5, 5, 6, 16])
        h3 = conv2d(h2, W3, b3, activation, padding='VALID')

        tf.summary.image('W3', W3)

    # S4
    with tf.name_scope('S4'):
        h4 = max_pool_2x2(h3)

    # C5
    with tf.name_scope('C5'):
        W5, b5 = conv_variables([5, 5, 16, 120])
        h5 = conv2d(h4, W5, b5, activation, padding='VALID')

        tf.summary.image('W5', W5)
        tf.summary.histogram('b5', b5)

        h_flatten = tf.reshape(h5, [-1, 120])

    # F6
    with tf.name_scope('C6'):
        W6, b6 = fc_variables([120, 84])
        h6 = tf.matmul(h_flatten, W6) + b6
        y_out = h6

        tf.summary.histogram('W6', W6)
        tf.summary.histogram('b6', b6)

    loss = tf.reduce_mean( 
                tf.nn.softmax_cross_entropy_with_logits(
                            labels=y_, logits=y_out)
                )

    predict = tf.equal(tf.argmax(y_out), tf.argmax(y_))
    accuracy = tf.reduce_mean(tf.cast(predict, dtype))

    return (x, y_), loss, accuracy, tf.train.Saver()
