# -*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant([1, 2, 3], dtype = tf.int32)
b = tf.expand_dims(a, 1)
# (3, 1)
print(b.shape)

c = b.shape.as_list()
# [3, 1], 数组
print(c)
