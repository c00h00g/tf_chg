# -*- coding:utf-8 -*-

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype = tf.int32)

# [3 6]
b = tf.reduce_max(a, 1)

# [4 5 6]
c = tf.reduce_max(a, 0)


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
