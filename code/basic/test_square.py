# -*- coding:utf=8 -*-
"""
对每一个元素计算平方
"""

import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype = tf.int32)
b = tf.square(a)

with tf.Session() as sess:
    # [[ 1  4  9]
    # [16 25 36]]
    print(sess.run(b))
