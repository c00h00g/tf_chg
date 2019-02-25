import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

labels = tf.constant([1, 2, 3])

one_hot_labels = tf.one_hot(labels, depth=5, dtype=tf.float32)
one_hot_labels_six = tf.one_hot(labels, depth=6, dtype=tf.float32)

with tf.Session() as sess:
    #[[0. 1. 0. 0. 0.]
    # [0. 0. 1. 0. 0.]
    # [0. 0. 0. 1. 0.]]
    print(sess.run(one_hot_labels))

    #[[0. 1. 0. 0. 0. 0.]
    # [0. 0. 1. 0. 0. 0.]
    # [0. 0. 0. 1. 0. 0.]]
    print(sess.run(one_hot_labels_six))
