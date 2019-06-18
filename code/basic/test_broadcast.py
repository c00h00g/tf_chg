# -*- coding:utf-8 -*-

"""
≤‚ ‘π„≤•ª˙÷∆
"""

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype = tf.float32)
print(a)

b = tf.constant([[[0.5, 0.5]], [[0.6, 0.6]]], dtype = tf.float32)
print(b)

c = a * b
print(c)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    
