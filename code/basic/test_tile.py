# -*- coding:utf-8 -*-

"""
纬度扩展
tile : 平铺的意思
定义:

tile(
    input,     #输入
    multiples, #同一维度上复制的次数
    name=None)

注意：复制的纬度必须和输入一致

技巧: 把对应维度乘以tile的维度得到新的维度
"""

import tensorflow as tf

# 1 * 2 * 3
a = tf.constant([[[1, 2, 3], [4, 5, 6]]], dtype = tf.int32)

#不进行复制
b = tf.tile(a, [1, 1, 1])

#[[[1 2 3]
#  [4 5 6]]
#
#  [[1 2 3]
#   [4 5 6]]]
c = tf.tile(a, [2, 1, 1])

#[[[1 2 3]
#  [4 5 6]
#  [1 2 3]
#  [4 5 6]]]
d = tf.tile(a, [1, 2, 1])

#[[[1 2 3 1 2 3]
#  [4 5 6 4 5 6]]]
e = tf.tile(a, [1, 1, 2])

#[[[1 2 3 1 2 3]
#  [4 5 6 4 5 6]]
#
#  [[1 2 3 1 2 3]
#  [4 5 6 4 5 6]]]
f = tf.tile(a, [2, 1, 2])

with tf.Session() as sess:
    print(sess.run(b))
    print('---->')

    print(sess.run(c))
    print('---->')

    print(sess.run(d))
    print('---->')

    print(sess.run(e))
    print('---->')

    print(sess.run(f))
    print('---->')
