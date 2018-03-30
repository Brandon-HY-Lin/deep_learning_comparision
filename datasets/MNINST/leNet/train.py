import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.MNIST import input_data

# read MNIST data set
# start session
#   init global variables
#   merge all summaries
#   start training
#       sample summary
#       train using mini-batch
# save graph

    train_step = tf.train.GradientDescentOptimizer(FLAG.lr).minimize(loss)

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())

        for i in tqdm.tqdm(range(FLAGS.max_step)):
            batch = data_set.train.next_batch(FLAGS.batch_size) 
            x_batch = batch[0]
            y_batch = batch[1]


