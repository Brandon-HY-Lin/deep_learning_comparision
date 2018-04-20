# build graph
# start valid
#   start a session
#       restore session
#       sess.run cost and accuarcy
#       print cost and error_rate

import tensorflow as tf
import numpy as np

from model import build_model 


def test(X_data, Y_data, 
            activation,
            job_dir, device_name):

    with tf.Graph().as_default() as v_graph:
        (x, y_), _, cost, error_rate, saver = \
                    build_model(activation, \
                                is_learning=False, enable_bn = True, 
                                device_name = device_name) 

    with tf.Session(graph = v_graph) as sess:
        saver.restore(sess, job_dir)

        cost, acc = sess.run(fetches=[cost, error_rate],
                            feed_dict={x: X_data, y_: Y_data})

        print("Test error_rate: %g" % (acc))
        print("Test cost: %g" % (cost))


