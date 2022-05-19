# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

a = np.random.randint(0, 5, size=(3, 4))
a_t = tf.convert_to_tensor(a)

b = np.random.randint(0, 5, size=(3, 4))
b_t = tf.convert_to_tensor(b)

c = tf.greater(a_t, b_t)


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b_t))
    print(sess.run(c))
    

"""
[[3 0 3 4]
 [0 3 0 2]
 [3 1 2 2]]

[[0 3 1 4]
 [1 0 1 4]
 [2 4 1 3]]

[[ True False  True False]
 [False  True False False]
 [ True False  True False]]
"""
