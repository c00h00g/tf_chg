# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


a = np.random.randint(0, 10, (3, 5))
input_data = tf.convert_to_tensor(a, dtype=tf.int32)

b = np.random.randint(0, 10, (5, 4, 3))
gate = tf.convert_to_tensor(b, dtype=tf.int32)


# -1表示input_data忽略最后一个维度
# 0表示gate忽略第一维
# 即: 5 * 5 做点乘之后，再sum
# 最终长度为 3 * 4 * 3
c = tf.tensordot(input_data, gate, axes = [-1, 0])

with tf.Session() as sess:
    print(sess.run(input_data))
    print(sess.run(gate))
    print(sess.run(c))

"""
Outputs:
[[7 2 8 0 0]
 [8 6 2 1 9]
 [9 9 9 9 1]]

[[[4 3 8]
  [7 9 9]
  [2 4 4]
  [7 7 9]]

 [[3 1 1]
  [1 2 9]
  [3 0 6]
  [6 9 0]]

 [[2 8 5]
  [3 2 8]
  [6 5 5]
  [4 4 4]]

 [[3 6 3]
  [8 2 0]
  [3 7 1]
  [8 9 7]]

 [[4 6 8]
  [6 0 6]
  [8 1 3]
  [3 9 5]]]

[[[ 50  87  98]
  [ 75  83 145]
  [ 68  68  80]
  [ 93  99  95]]

 [[ 93 106 155]
  [130  90 196]
  [121  58 106]
  [135 208 132]]

 [[112 168 161]
  [177 135 240]
  [134 145 147]
  [228 270 185]]]
"""
