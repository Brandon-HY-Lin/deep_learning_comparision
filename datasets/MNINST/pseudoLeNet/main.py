import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data
import numpy as np
import tqdm
from build_graph import build_graph
from test import test
from train import train
import argparse

FLAGS=None

def main():
    # get data
    # (graph 1)
    # build graph
    # start training
    #   init global variables
    #   merge all summaries
    #   iterate training set
    #       if i % summary_period == 0 or i+1 == max_epochs
    #           run summary tensor
    #
    #       run train tensor
    #
    #   save model
    #
    # (graph 2)
    # build graph
    # start testing
    #   restore model
    #   eval accuracy of testing set

    data_set = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with tf.Graph().as_default() as g_train:
        with tf.device(FLAGS.device_name):
            (x, y_), loss, accuracy, saver = \
                        build_graph(is_learning=True, enable_bn=True) 

    with tf.Session(graph=g_train) as sess:
        train(x, y_, loss, FLAGS.lr, accuracy,
                data_set, FLAGS.batch_size, FLAGS.max_step, 
                FLAGS.summary_period, FLAGS.print_period,
                sess, saver, FLAGS.model_dir, FLAGS.training_log_dir)

    with tf.Graph().as_default() as g_test:
        with tf.device(FLAGS.device_name):
            (x, y_), loss, accuracy, saver = \
                        build_graph(is_learning=False, enable_bn=True)

    with tf.Session(graph=g_test) as sess:
        test(x, y_, accuracy, data_set, 
                saver, FLAGS.model_dir, sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./MNIST_data',
                        help='Directory for storing input data.')

    parser.add_argument('--model_dir', type=str,
                        default='./log/model',
                        help='Directory for saving/restoring model')

    parser.add_argument('--training_log_dir', type=str,
                        default='./log/',
                        help='Directory for storing training log')

    parser.add_argument('--max_step', type=int,
                        #default=20000,
                        default=300,
                        help="Max epochs to run trainer.")

    parser.add_argument('--batch_size', type=int,
                        default=50,
                        help="Batch size for batch normalization.")

    parser.add_argument('--lr', type=float,
                        default=1e-4,
                        help="Learning rate.")

    parser.add_argument('--device_name', type=str,
                        #default="/device:GPU:0",
                        default="/cpu:0",
                        help="Choose CPU or GPU.")

    parser.add_argument('--summary_period', type=int,
                        default=100,
                        #default=1,
                        help='Sampling period of summary')

    parser.add_argument('--print_period', type=int,
                        default=100,
                        #default=1,
                        help='Period of printing log on bash')

    FLAGS, unparsed = parser.parse_known_args()

    main()
