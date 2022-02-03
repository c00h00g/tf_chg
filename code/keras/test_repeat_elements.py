# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


a = np.random.randint(0, 3, size=(2, 3, 4))
a_t = tf.convert_to_tensor(a)

# 三个参数
# x: 输入
# rep: 重复次数 
# axis：沿哪个轴
b = tf.keras.backend.repeat_elements(a_t, 2, axis=-1)
c = tf.keras.backend.repeat_elements(a_t, 2, axis=1)
d = tf.keras.backend.repeat_elements(a_t, 2, axis=0)


with tf.Session() as sess:
    print('chg a_t is -------------------->')
    print(sess.run(a_t))

    print('chg axis = 2 is -------------------->')
    print(sess.run(b))

    print('chg axis = 1 is -------------------->')
    print(sess.run(c))

    print('chg axis = 0 is -------------------->')
    print(sess.run(d))

"""
chg a_t is -------------------->
[[[2 0 1 1]
  [1 0 0 0]
  [0 2 2 2]]

 [[0 2 1 1]
  [0 0 1 0]
  [0 1 0 2]]]

chg axis = 2 is -------------------->
[[[2 2 0 0 1 1 1 1]
  [1 1 0 0 0 0 0 0]
  [0 0 2 2 2 2 2 2]]

 [[0 0 2 2 1 1 1 1]
  [0 0 0 0 1 1 0 0]
  [0 0 1 1 0 0 2 2]]]

chg axis = 1 is -------------------->
[[[2 0 1 1]
  [2 0 1 1]
  [1 0 0 0]
  [1 0 0 0]
  [0 2 2 2]
  [0 2 2 2]]

 [[0 2 1 1]
  [0 2 1 1]
  [0 0 1 0]
  [0 0 1 0]
  [0 1 0 2]
  [0 1 0 2]]]

chg axis = 0 is -------------------->
[[[2 0 1 1]
  [1 0 0 0]
  [0 2 2 2]]

 [[2 0 1 1]
  [1 0 0 0]
  [0 2 2 2]]

 [[0 2 1 1]
  [0 0 1 0]
  [0 1 0 2]]

 [[0 2 1 1]
  [0 0 1 0]
  [0 1 0 2]]]
"""
