import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data
import numpy as np
import tqdm
from build_graph import build_graph

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
        training(loss, FLAGS.lr, data_set,
                FLAGS.batch_size, FLAGS.max_step,
                sess, FLAGS.model_dir, FLAGS.training_log_dir)

    with tf.Graph().as_default() as g_test:
        with tf.device(FLAGS.device_name):
            (x, y_), loss, accuracy, saver = \
                        build_graph(is_learning=False, enable_bn=True)

    with tf.Session(graph=g_test) as sess:
        testing(accuracy, data_set,
                sess, FLAGS.model_dir, FLAGS.testing_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./MNIST_data',
                        help='Directory for storing input data.')

    parser.add_argument('--model_dir', type=str,
                        default='./model/model-',
                        help='Directory for saving/restoring model')

    parser.add_argument('--training_log_dir', type=str,
                        default='./log/train-',
                        help='Directory for storing training log')

    parser.add_argument('--testing_log_dir', type=str,
                        default='./log/test-',
                        help='Directory for storing testing log')

    parser.add_argument('--max_step', type=int,
                        #default=20000,
                        default=200,
                        help="Max epochs to run trainer.")

    parser.add_argument('--batch_size', type=int,
                        default=60,
                        help="Batch size for batch normalization.")

    parser.add_argument('--lr', type=float,
                        default=0.5,
                        help="Learning rate.")

    parser.add_argument('--device_name', type=str,
                        #default="/device:GPU:0",
                        default="/cpu:0",
                        help="Choose CPU or GPU.")

    FLAGS, unparsed = parser.parse_known_args()

    main()
