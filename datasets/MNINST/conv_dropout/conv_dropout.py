import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# train the train data with prob = 0.5
#   build training optimizer
#       build cross entropy graph
#       build 1-layer nn with dropout input
#       build 2-laywers conv with ReLu activation
#   evaluate optimizer
# evaluate the test data with prob = 1.0
#   build accuracy graph
#   build prediction graph

#device_name = '/device:GPU:0'
device_name = '/cpu:0'

def weight_variable(shape, stddev=0.1):
    init = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(init)

def bias_variable(shape, value=0.1):
    init = tf.constant(value, shape=shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, 
                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
    dtype = tf.float32

    PATH_DOWNLOADE_DATA = 'MNIST_data/'
    mnist = input_data.read_data_sets(PATH_DOWNLOADE_DATA, one_hot=True)

    layers_size = {
        'fc1': [7*7*64, 1024],
        'fc2': [1024, 10],
    }

    #HWCC
    filters_shape = {
        'conv1': [5, 5, 1, 32],
        'conv2': [5, 5, 32, 64],
    }

    x = tf.placeholder(dtype, [None, 784])
    y_ = tf.placeholder(dtype, [None, 10])

    # convert 1d image to 2d image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # activation function
    f = tf.nn.relu

    with tf.device(device_name):
        # define conv layer 1
        #   conv
        #   max pool
        W_conv1 = weight_variable(filters_shape['conv1'])
        b_conv1 = bias_variable([filters_shape['conv1'][-1]])

        h_conv1 = f(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # define conv layer 2
        #   conv
        #   max pool
        W_conv2 = weight_variable(filters_shape['conv2'])
        b_conv2 = bias_variable([filters_shape['conv2'][-1]])

        h_conv2 = f(conv2d(h_pool1, W_conv2) + b_conv2)  
        h_pool2 = max_pool_2x2(h_conv2) 

        # define hidden layer
        W_fc1 = weight_variable(layers_size['fc1'])
        b_fc1 = bias_variable([layers_size['fc1'][-1]])

        h_pool2_flat = tf.reshape(h_pool2, [-1, layers_size['fc1'][0]])
        h_fc1 = f(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # define logits with dropout inputs
        W_fc2 = weight_variable(layers_size['fc2'])
        b_fc2 = bias_variable([layers_size['fc2'][-1]])

        keep_prob = tf.placeholder(dtype)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # define cross_entropy graph
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        )

        # define train optimizer tensor
        lr=1e-4
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

        # define accuracy
        prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, dtype))

    # size of the training data set is 55k
    num_epochs = 20000 #20k
    batch_sz = 50
    keep_prob_train = 0.5

    # init shared variable
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    print_period = 100

    for i in range(num_epochs):
        batch = mnist.train.next_batch(batch_sz)
        
        if i % print_period == 0:
            # print current training accuracy
            train_accuracy = sess.run(fetches=[accuracy],
                    feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}
                )

            print("step %d, accuracy %g" % (i, train_accuracy[0]))

        sess.run(fetches=train_step,
            feed_dict={x: batch[0], y_: batch[1], 
                        keep_prob: keep_prob_train}
        )

    # print test accuracy
    acc = sess.run(fetches=[accuracy],
        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    
    print(
        "test accuracy %g" % acc[0]
    )

if __name__ == '__main__':
    main()
