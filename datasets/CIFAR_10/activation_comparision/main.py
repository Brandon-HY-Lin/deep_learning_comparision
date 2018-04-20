# read CIFAR-10 dataset
# start training
# start validating

# define bash arguments

import tensorflow as tf
import numpy as np

from include.data import get_data_set
from train import train
from test import test

import argparse


FLAGS=None

def main():
    activations = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh}
    #activations = {'tanh': tf.nn.tanh}
    #activations = {'relu': tf.nn.relu}
        
    X_train, Y_train, label_names = get_data_set('train')
    X_test, Y_test, _ = get_data_set('test')

    for name_activation, activation in activations.items():
        print("Starting training %s:" % (name_activation))

        tf.reset_default_graph()
    
        job_dir = FLAGS.job_dir + name_activation + '/'
    
        train(X_train, Y_train, 
                FLAGS.learning_rate, FLAGS.train_steps,
                activation, FLAGS.train_batch_size, 
                job_dir,
                FLAGS.device_name, FLAGS.log_period)
    
        test(X_test, Y_test, 
                activation,
                job_dir, FLAGS.device_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # num of training cases: 50k
    # it takes: ~390 steps to simulate full-batch if batch size=128
    # AlexNet paper shows it needs 35 full-batch steps to achieve 25% error.
    parser.add_argument('--train_steps', type=int,
                        default=15000,
                        #default=40,
                        help='Max epochs to run trainer.')

    # diverge: 0.1, 10e-3
    parser.add_argument('--learning_rate', type=float,
                        default=1.05*10e-4,
                        help='Learning rate.')

    parser.add_argument('--train_batch_size', type=int,
                        default=128,
                        help='Batch size for training.')

    parser.add_argument('--job_dir', type = str,
                        default='./log/',
                        help=   """\
                                The directory where the model will be 
                                stored.\
                                """)

    parser.add_argument('--device_name', type=str,
                        default='/device:GPU:0',
                        #default='/cpu:0',
                        help='Choose CPU or GPU.')

    # if batch size= 128, a full-batch equals 390 mini-batch steps
    parser.add_argument('--log_period', type=int,
                        default=100,
                        #default=5,
                        help='Period for writing log.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
