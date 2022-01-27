
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


def test_1(sess):
    #ouptus
    #[[0 0]
    # [1 1]
    # [0 1]]
    #[[0 1]
    # [1 2]
    # [2 1]]
    #[[0]
    # [3]
    # [1]]
    # batch保持不变, 第二维进行x * x^T 维度是1
    a = np.random.randint(0, 3, size=(3, 2))
    a_t = tf.convert_to_tensor(a)

    b = np.random.randint(0, 3, size=(3, 2))
    b_t = tf.convert_to_tensor(b)

    # a_t : (3, 2)
    # b_t : (3, 2)
    # output: 3 * 1
    c = tf.keras.backend.batch_dot(a_t, b_t, axes=1)
    print(sess.run(a_t))
    print(sess.run(b_t))
    print(sess.run(c))

def test_2(sess):
    x_batch = tf.convert_to_tensor(np.random.randint(0, 3, size=(2, 2, 1)))
    y_batch = tf.convert_to_tensor(np.random.randint(0, 3, size=(2, 3, 2)))
    xy_batch_dot = tf.keras.backend.batch_dot(x_batch, y_batch, axes=(1, 2))
    print(sess.run(x_batch))
    print(sess.run(y_batch))
    print(sess.run(xy_batch_dot))



with tf.Session() as sess:
   print('chg test_1 result is ---------------------------->') 
   test_1(sess)
   print('chg test_2 result is ---------------------------->') 
   test_2(sess)


"""
chg test_1 result is ---------------------------->
[[1 0]
 [2 2]
 [2 1]]
[[0 2]
 [1 1]
 [1 0]]
[[0]
 [4]
 [2]]

chg test_2 result is ---------------------------->
[[[2]
  [1]]

 [[0]
  [0]]]

[[[0 0]
  [0 0]
  [2 1]]

 [[0 2]
  [0 2]
  [2 1]]]
[[[0 0 5]]

 [[0 0 0]]]
"""







