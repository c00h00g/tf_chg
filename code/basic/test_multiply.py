# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np


a = np.random.randint(1, 3, size=(3, 5))
a_t = tf.convert_to_tensor(a)

b = np.random.randint(1, 3, size=(3, 5))
b_t = tf.convert_to_tensor(b)

c = tf.multiply(a_t, b_t)

with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b_t))
    print(sess.run(c))

"""
[[2 2 1 1 1]
 [1 1 1 2 2]
 [2 1 1 2 2]]

[[1 2 1 2 2]
 [2 1 2 2 1]
 [1 2 1 1 1]]

[[2 4 1 2 2]
 [2 1 2 4 2]
 [2 2 1 2 2]]
"""
