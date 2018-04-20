# reshape x to (-1, 32, 32, 3)
# C1: 
#   conv1:  F=5, S=1, SAME, #out channel: 64
#       outputs: 32x32x64
#       params: 5x5x3x64 + 64 = 4,864
#
# S1:
#   pool: F=3, S=2, SAME
#       outputs: 16x16x64
#
# C2:
#   conv2: F=5, S=1, SAME, #out channel: 64
#       outputs: 16x16x64
#       params: 5x5x64x64 + 64 = 102,400
#
# S2:
#   pool: F=3, S=2, SMAE
#       outputs: 8x8x64
#
# F3:
#   FC3: (8x8x64)x384
#       params= (8x8x64)x384 + 384 = 1,572,864
#
# F4:
#   FC4: 384x192
#       params= 384x192 + 192 = 73,920
#
# O5:
#   output5: 192x10
#       params = 192x10 + 10= 1,930
#
# total params = 4,864 + 102,400 + 1,572,864 + 73,920 + 1,930
#               = 1,764,048

import tensorflow as tf
import numpy as np

dtype = tf.float32

def create_placeholders(n_H0, n_W0, n_C0, n_y, dtype=tf.float32):
    x = tf.placeholder(dtype, shape=[None, n_H0*n_W0*n_C0], name='x')
    y_ = tf.placeholder(dtype, shape=[None, n_y])

    return (x, y_)

def create_cost(y_out, labels):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=y_out, labels=labels
            ))

    return cost

def create_error_rate(y_out, labels):
    predict = \
            tf.equal(tf.argmax(y_out, axis=1), tf.argmax(labels, axis=1))

    return 1.0 - tf.reduce_mean(tf.cast(predict, tf.float32))

def get_variables(shape, dtype=tf.float32):
    W_init = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(W_init, dtype=dtype)

    b_init = tf.constant(0.1, shape=[shape[-1]])
    b = tf.Variable(b_init, dtype=dtype)

    return W, b

def conv2d(X, W, b, activation, padding='SAME'):
    Z = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding=padding) + b
    A = activation(Z)

    return A

def fully_connected(X, W, b):
    Z = tf.matmul(X, W) + b
    return Z

def max_pool_2x2(X, F=3, S=2):
    Z = tf.nn.max_pool(X, ksize=[1, F, F, 1], strides=[1, S, S, 1], padding='SAME')

    return Z

def forward_propagation(X_img, activation):
    with tf.variable_scope('conv1'):
        W1, b1 = get_variables([5, 5, 3, 64])
        A1 = conv2d(X_img, W1, b1, activation)

        M1 = max_pool_2x2(A1)


    with tf.variable_scope('conv2'):
        W2, b2 = get_variables([5, 5, 64, 64])
        A2 = conv2d(M1, W2, b2, activation)

        M2 = max_pool_2x2(A2)

        M2_flatten = tf.layers.Flatten()(M2)
        #M2_flatten = tf.reshape(M2, [-1, 8*8*64])

    with tf.variable_scope('fc3'):
        W3, b3 = get_variables([8*8*64, 384])
        Z3 = fully_connected(M2_flatten, W3, b3)
        A3 = activation(Z3)

    with tf.variable_scope('fc4'):
        W4, b4 = get_variables([384, 192])
        Z4 = fully_connected(A3, W4, b4)
        A4 = activation(Z4)

    with tf.variable_scope('fc5'):
        W5, b5 = get_variables([192, 10])
        Z5 = fully_connected(A4, W5, b5)
    
        y_out = Z5

    return y_out

def build_model(activation, is_learning, enable_bn, device_name):
    n_H0 = 32
    n_W0 = 32
    n_C0 = 3
    n_y = 10

    (x, y_) = create_placeholders(n_H0, n_W0, n_C0, n_y)

    with tf.device(device_name):
        x_img = tf.reshape(x, [-1, n_H0, n_W0, n_C0])

        y_out = forward_propagation(x_img, activation)
        cost = create_cost(y_out, y_)
        error_rate = create_error_rate(y_out, y_)

    with tf.variable_scope('cost'):
        tf.summary.scalar('cost', cost)

    with tf.variable_scope('error_rate'):
        tf.summary.scalar('error_rate', error_rate)

    return (x, y_), y_out, cost, error_rate, tf.train.Saver()
