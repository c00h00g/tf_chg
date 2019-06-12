# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


a = tf.constant([1, 2, 3])
b = tf.constant([1, 2, 3])

# 1 * 3
c = tf.constant([[1, 2, 3]])

# 3 * 1
d = tf.constant([[1], [2], [3]])

# (3, 3, 2)
e = tf.constant([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]], [[7, 7], [8, 8], [9, 9]]])
print(e.shape)

# (3, 2)
f = tf.constant([[2, 2], [3, 3], [4, 4]])
print(f.shape)

# (3, 3, 2)
# f和每个batch乘
g = tf.multiply(e, f)
print(g.shape)

with tf.Session() as sess:
    #[1 4 9]
    print(sess.run(tf.multiply(a, b)))

    #[1 4 9]
    #矩阵点乘可以直接用乘号
    print(sess.run(a * b))

    #[[14]]
    print(sess.run(tf.matmul(c, d)))

    #[[[ 2  2]
    #  [ 6  6]
    #  [12 12]]

    #  [[ 8  8]
    #   [15 15]
    #   [24 24]]

    #  [[14 14]
    #   [24 24]
    #   [36 36]]]
    print(sess.run(g))
