# -*- coding:utf-8 -*-

import tensorflow as tf


a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype = tf.int32)

# (2, 1, 3)
b = tf.expand_dims(a, 1)

# 3 : 表示总共几个纬度
print(b.shape.ndims)

