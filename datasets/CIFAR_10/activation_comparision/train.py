# build graph
# start train
#   start session
#       merge summary
#       create FileWriter
#       add session.graph to FileWriter
#
#       intialize global variabels
#
#       for each epochs (tqdm iterator)
#           if print_priod == 0 or (print_period) >= train_steps:
#               run cost
#               run summary
#               add summary to FileWriter
#           run optimizer
#
#       save model to job_dir
#
#       (no need to close session because of context manager)

import tensorflow as tf
import numpy as np
import tqdm

from model import build_model

def train(X_data, Y_data, 
            learning_rate, train_steps, 
            activation, batch_size, 
            job_dir, 
            device_name, log_period):

    with tf.Graph().as_default() as t_graph:
        (x, y_), _, cost, accuracy, saver = \
                build_model(activation, is_learning=True, 
                            enable_bn=True, device_name = device_name)

    with tf.Session(graph=t_graph) as sess:
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(job_dir)
        writer.add_graph(sess.graph)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        sess.run(tf.global_variables_initializer())

        for i in tqdm.tqdm(range(train_steps)):
            i_start = i * batch_size
            i_end = (i+1) * batch_size
    
            x_batch = X_data[i_start: i_end]
            y_batch = Y_data[i_start: i_end]
    
            # tqdm starts from 1
            if (i % log_period) == 0 or i >= train_steps:
                s = sess.run( fetches = merged_summary, 
                            feed_dict={x: x_batch, y_: y_batch})
    
                writer.add_summary(s, i)

                acc, c = sess.run( fetches=[accuracy, cost],
                            feed_dict={x: x_batch, y_: y_batch})

                print('Step %d, accuracy = %g, cost = %g' % (i, acc, c))
    
    
            sess.run(fetches=optimizer,
                    feed_dict={x: x_batch, y_: y_batch})
    
    
        saver.save(sess, job_dir)
