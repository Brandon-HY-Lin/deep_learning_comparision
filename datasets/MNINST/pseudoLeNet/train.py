# merge all summaries
# flush Graph to file
# init global variables
# start training
#   sample summary and flush it
#   train using mini-batch
# save graph

import tensorflow as tf
import numpy as np
import tqdm

def train(x, y_, loss, lr, accuracy,
            data_set, batch_size, max_step, 
            summary_period, print_period,
            sess, saver, model_dir, training_log_dir):

    collection_train='train'
    collection_all='all'
    summary_train = tf.summary.merge_all(collection_train)
    summary_all = tf.summary.merge_all(collection_all)

    merged_summary = tf.summary.merge([summary_train, summary_all])

    writer = tf.summary.FileWriter(training_log_dir)

    # show Graph on the web browser
    writer.add_graph(sess.graph)

    #train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    sess.run(tf.global_variables_initializer())

    for i in tqdm.tqdm(range(max_step)):
        batch = data_set.train.next_batch(batch_size) 
        #x_batch = batch[0]
        #y_batch = batch[1]

        encoded = np.argmax(batch[1], 1)
        #index_digit = np.logical_or(encoded== 8, encoded== 9)
        index_digit = (encoded== 7)
        x_batch = batch[0][index_digit][:]
        y_batch = batch[1][index_digit][:]

        # tqdm starts from 1
        if i % summary_period == 0 or i >= max_step:
            s = sess.run(fetches=merged_summary,
                    feed_dict={x: x_batch, y_: y_batch})

            writer.add_summary(s, i)

        if i % print_period == 0 or i >= max_step:
            acc = sess.run(fetches=accuracy,
                    feed_dict={x: x_batch, y_: y_batch})

            print("Step %d, accuracy: %g" % (i, acc))


        sess.run(fetches=train_step,
                feed_dict={x: x_batch, y_: y_batch})

    saver.save(sess, model_dir)
