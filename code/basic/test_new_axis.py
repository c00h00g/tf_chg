# -*- coding:utf-8 -*-
# 增加一个维度

import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = a[:, :, tf.newaxis]

print("a shape is -------->")
print(a.shape)

print("b shape is -------->")
print(b.shape)


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))

#a shape is -------->
#(2, 3)
#b shape is -------->
#(2, 3, 1)
#
#[[1 2 3]
# [4 5 6]]
#[[[1]
#  [2]
#  [3]]
#
# [[4]
#  [5]
#  [6]]]
