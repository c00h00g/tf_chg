# -*- coding:utf-8 -*-

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


a = tf.constant([1, 2, 3])
a = tf.expand_dims(a, 1)
#(3, 1)
print(a.shape)

b = tf.constant([[1, 1], [2, 2], [3, 3]])
#(3, 2)
print(b.shape)

with tf.Session() as sess:
    #[[1 1]
    # [4 4]
    # [9 9]]
    print(sess.run(a * b))


