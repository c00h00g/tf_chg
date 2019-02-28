import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.Variable([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
index_a = tf.Variable([0, 2])

b = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
index_b = tf.Variable([2, 4, 6, 8])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #[[ 1  2  3  4  5]
    # [11 12 13 14 15]]
    print(sess.run(tf.gather(a, index_a)))

    # [3 5 7 9]
    print(sess.run(tf.gather(b, index_b)))
