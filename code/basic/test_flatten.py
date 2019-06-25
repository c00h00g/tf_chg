# -*- coding:utf-8 -*-

"""
保持第一个维度不变，后面几个维度展开成一个维度
"""
import tensorflow as tf

#(2, 3, 2)
a = tf.constant([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]], dtype = tf.int32)
print(a.shape)

#(2, 6) == (2, 3, 2) --> (2, 3 * 2)
b = tf.layers.flatten(a)
print(b.shape)

with tf.Session() as sess:
    # [[1 1 2 2 3 3]
    # [4 4 5 5 6 6]]
    print(sess.run(b))
