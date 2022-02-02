# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

a = np.random.randint(0, 3, size=(2, 3, 4))
a_t = tf.convert_to_tensor(a)

b = tf.keras.backend.permute_dimensions(a_t, (0, 2, 1))


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))


"""
[[[2 1 2 2]
  [0 1 1 1]
  [2 2 2 2]]

 [[0 0 0 0]
  [0 2 0 2]
  [2 2 1 2]]]

转置后结果
[[[2 0 2]
  [1 1 2]
  [2 1 2]
  [2 1 2]]

 [[0 0 2]
  [0 2 2]
  [0 0 1]
  [0 2 2]]]
"""
