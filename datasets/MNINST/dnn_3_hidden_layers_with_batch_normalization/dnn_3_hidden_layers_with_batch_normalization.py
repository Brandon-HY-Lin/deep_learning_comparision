########################################
#
# Reproduced work of Ioffe and Szegedy's paper
#   Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (https://arxiv.org/abs/1502.03167)
#
########################################

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data, mnist
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os

dtype = tf.float32

def get_path_saved_model(prefix, suffix):
    return prefix + suffix + "/tmp-save"

def mkdir_if_not_exits(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
  

def weight_variable(shape, stddev=0.1):
    init = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(init, dtype=dtype)

def batch_norm_wrapper(x, is_training, enable_bn, decay=0.999):
    epsilon = 1e-8

    size = x.get_shape()[-1]

    run_mean    = tf.Variable(tf.zeros([size]), 
                                dtype=dtype, trainable=False)

    run_var     = tf.Variable(tf.ones([size]), 
                                dtype=dtype, trainable=False)

    offset      = tf.Variable(tf.zeros([size]), dtype=dtype)
    scale       = tf.Variable(tf.ones([size]), dtype=dtype)

    if enable_bn == True:
        if is_training == True:
            batch_mean, batch_var = tf.nn.moments(x, axes=[0])

            update_run_mean = tf.assign(
                                run_mean, 
                                decay*run_mean + (1-decay)*batch_mean
                    )
            update_run_var = tf.assign(
                        run_var,
                        decay*run_var + (1-decay)*batch_var
                    )

            with tf.control_dependencies([update_run_mean, update_run_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var,
                        offset=offset, scale=scale,
                        variance_epsilon=epsilon)

        else:
            return tf.nn.batch_normalization(x, run_mean, run_var,
                        offset=offset, scale=scale, 
                        variance_epsilon=epsilon)
    else:
        return x 

def build_graph(is_training, enable_bn):
    # build graph with 3 hidden layers with 100 sigmoid activations each.
    # variables are initialized to small Gaussian values

    shapes={'h1': [mnist.IMAGE_PIXELS, 100],
            'h2': [100, 100],
            'h3': [100, 100],
            'output': [100, mnist.NUM_CLASSES]}

    activation = tf.nn.sigmoid

    x = tf.placeholder(dtype, shape=[None, mnist.IMAGE_PIXELS])
    y_ = tf.placeholder(dtype, shape=[None, mnist.NUM_CLASSES])

    # 1st hidden layer
    W1 = weight_variable(shapes['h1'])
    z1 = tf.matmul(x, W1)
    bn1 = batch_norm_wrapper(z1, is_training, enable_bn)
    l1 = activation(bn1)

    # 2nd hidden layer
    W2 = weight_variable(shapes['h2'])
    z2 = tf.matmul(l1, W2)
    bn2 = batch_norm_wrapper(z2, is_training, enable_bn)
    l2 = activation(bn2)

    # 3rd hidden layer
    W3 = weight_variable(shapes['h3'])
    z3 = tf.matmul(l2, W3)
    bn3 = batch_norm_wrapper(z3, is_training, enable_bn)
    l3 = activation(bn3)

    # output layer
    W_out = weight_variable(shapes['output'])
    y = tf.matmul(l3, W_out) 

    loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                        logits=y)
            )

    lr = 0.5
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, dtype))

    return (x, y_), train_step, accuracy, y, tf.train.Saver()

def main():
    # iterate settings: {'with_bn':True, 'without_bn':False}
    #   start training
    #     build graph
    #     start session
    #       init global variables
    #       feed the data set
    #           if print_period
    #               save batch accuracy into history
    #
    #           train the batch data
    #       save the model
    #   start testing
    #     reset the default graph
    #     build the graph
    #     start session
    #       init global variable
    #       restore model
    #       print accuracy
    #
    # plot test accuracy history

    # {title: enable batch normalization}
    experiment_settings = {'with_bn': True, 'without_bn': False}

    FLAGS_ = {  'batch_sz': 60,
                'max_epochs': 50000,
                'print_period': 100,
                'path_data_set': 'MNIST_data/',
                'path_saved_model': './tmp/',
            }

    history_acc_ = defaultdict(list)

    data_set = input_data.read_data_sets(FLAGS_['path_data_set'], 
                                        one_hot=True)

    for title, enable_bn in experiment_settings.items():
        # start training
        # build graph
        (x, y_), train_step, accuracy, _, saver = build_graph( 
                            is_training=True, 
                            enable_bn=enable_bn
                        )
        path_saved_model = get_path_saved_model(
                                    FLAGS_['path_saved_model'], title)

        mkdir_if_not_exits(path_saved_model)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in tqdm.tqdm(xrange(FLAGS_['max_epochs'])):
                batch = data_set.train.next_batch(FLAGS_['batch_sz'])

                if i % FLAGS_['print_period'] == 0 or \
                    (i+1) == FLAGS_['max_epochs'] :

                    acc = sess.run(fetches=[accuracy], 
                        feed_dict={ x: data_set.test.images, 
                                    y_: data_set.test.labels})

                    history_acc_[title].append(acc[0])

                # start training
                sess.run(fetches=[train_step], 
                    feed_dict={x: batch[0], y_: batch[1]})

            print("training accuracy of %s: %g" % 
                    (title, history_acc_[title][-1]) )

            # save model
            saver.save(sess, path_saved_model)

        # start testing
        tf.reset_default_graph()

        (x, y_), _, accuracy, y, saver = build_graph(is_training=False,
                                                    enable_bn=enable_bn)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, path_saved_model)
            acc = sess.run(fetches=[accuracy], 
                        feed_dict={x: data_set.test.images,
                                    y_: data_set.test.labels})

            print("testing accuracy of %s: %g" % 
                    (title, acc[0]))

        # reset default graph before new iteration
        tf.reset_default_graph()
                
    # end of iteration

    # plot test accuracy history
    for title, _ in experiment_settings.items():
        acc = np.array(history_acc_[title])
        plt.plot(acc, label=title)
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
