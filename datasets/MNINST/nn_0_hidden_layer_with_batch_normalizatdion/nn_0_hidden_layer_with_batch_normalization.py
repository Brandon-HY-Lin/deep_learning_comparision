import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import tqdm

dtype = tf.float32

def weight_variable(shape, stddev=0.1):
    init = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(init, dtype=dtype)

def bias_variable(shape, value=0.1):
    init = tf.constant(value=value, shape=shape)
    return tf.Variable(init, dtype=dtype)

def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    # calculate running mean and var
    # update running mean and var
    # set up a depenceies barrier
    # batch normalization

    variance_epsilon = 1e-8
    size = inputs.get_shape()[-1]

    scale = tf.Variable(tf.ones([size]))
    offset = tf.Variable(tf.zeros([size]))

    r_mean = tf.Variable(tf.zeros([size]), trainable=False)
    r_var = tf.Variable(tf.ones([size]), trainable=False)

    if is_training == True:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        assign_mean = tf.assign(r_mean, 
                                decay*r_mean + (1-decay)*batch_mean)
                        
        assign_var = tf.assign(r_var, 
                               decay*r_var + (1-decay)*batch_var)

        with tf.control_dependencies([assign_mean, assign_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var,
                        offset=offset, scale=scale, 
                        variance_epsilon=variance_epsilon)
    else:
        return tf.nn.batch_normalization(inputs, r_mean, r_var,
                        offset=offset, scale=scale, 
                        variance_epsilon=variance_epsilon)
                        

def build_graph(is_training):
    # build NN with 1 hidden layer. Activation function is Relu
    # return placeholders(x, y_) ,training tensor, accuracy tensor, output of last layer before softmax, tf.train.saver() 

    x = tf.placeholder(dtype, [None, mnist.IMAGE_PIXELS])
    y_ = tf.placeholder(dtype, [None, mnist.NUM_CLASSES])

    # build last layer
    W = weight_variable(shape=[mnist.IMAGE_PIXELS, mnist.NUM_CLASSES])
    b = bias_variable(shape=[mnist.NUM_CLASSES])
    y_fc1 = tf.matmul(x, W) + b

    y = batch_norm_wrapper(y_fc1, is_training)

    # define cross entropy
    xentropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #lr = 1e-4
    #train_step = tf.train.AdamOptimizer(lr).minimize(xentropy)
    lr = 0.5
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(xentropy)

    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, dtype))

    return (x, y_), train_step, accuracy, y, tf.train.Saver()

def main():
    # start train
    # build training graph
    # init global variables
    # get MNIST data
    # start batch training
    #   store training accuracy
    # save model

    PATH_SAVED_MODEL='./tmp/temp-nn-save'

    # build training graph
    (x, y_), train_step, accuracy, _, saver = build_graph(is_training=True)

    # get MNINST data
    data_set = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # start batch training
    num_train_epochs = 1000
    batch_sz = 100
    print_period = 50

    acc = []

    with tf.Session() as sess:
        # init global variables
        sess.run(tf.global_variables_initializer())
    
    
        # show progress bar in the bash-shell terminal
        for i in tqdm.tqdm(xrange(num_train_epochs)):
            batch = data_set.train.next_batch(batch_sz)
            if i % print_period == 0:
                v_accuracy = sess.run(fetches=[accuracy], 
                        feed_dict={x: data_set.test.images, y_: data_set.test.labels})
                acc.append(v_accuracy)
    
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        saved_model = saver.save(sess, PATH_SAVED_MODEL)

    print("Final training accuracy:", acc[-1])

    # start test
    # reset graph
    # build test graph
    # init global variables
    # restore model from file

    # reset graph
    tf.reset_default_graph()

    sess = tf.Session()

    # build test graph
    (x, y_), _, accuracy, y, saver = build_graph(is_training=False)


    # init global variables
    sess.run(tf.global_variables_initializer())

    # restore model from file
    saver.restore(sess, PATH_SAVED_MODEL)

    v_accuracy = sess.run(fetches=[accuracy], 
        feed_dict={x: data_set.test.images, y_: data_set.test.labels})

    print("Accuracy: %g" % (v_accuracy[0]))

if __name__ == '__main__':
    main()
