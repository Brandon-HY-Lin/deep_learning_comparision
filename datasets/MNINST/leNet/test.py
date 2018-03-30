# restore model
# run accuracy

def test(accuracy, data_set, 
            saver, model_dir, sess):
    
    saver.restore(sess, model_dir)
    print("Model restored")

    acc = sess.run(fetches=accuracy, 
                    feed_dict={x: data_set.test.images, 
                                y_: data_set.test.labels})

    print("Testing Accuracy: %g" % (acc))
