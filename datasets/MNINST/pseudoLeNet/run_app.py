import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data
from build_graph import build_graph

def main():
    model_dir = './log/model'
    device_name="/cpu:0"
    log_dir = './log/'
    data_dir = 'MNIST_data'

    data_set = input_data.read_data_sets(data_dir, one_hot=True)

    # build graph
    with tf.Graph().as_default() as g:
        with tf.device(device_name):
            (x, y_), loss, accuracy, saver = \
                    build_graph(is_learning=False, enable_bn=False)

    # run session
    with tf.Session(graph=g) as sess:
        # restore model
        saver.restore(sess, model_dir)

        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(sess.graph)

        merged_summary = tf.summary.merge([tf.summary.merge_all('all'),
                                        tf.summary.merge_all('layers')])

        s = sess.run(fetches=merged_summary, 
                        feed_dict={x: [data_set.test.images[0]], 
                            y_: [data_set.test.labels[0]]})
        #   write log
        writer.add_summary(s, 0)

if __name__ == '__main__':
    main()
