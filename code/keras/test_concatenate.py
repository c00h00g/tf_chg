# -*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[7, 8, 9], [10, 11, 12]])



c = tf.keras.layers.Concatenate(axis=0)([a, b])
d = tf.keras.layers.Concatenate(axis=1)([a, b])


with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))

"""
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]

[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
"""
