# -*- coding:utf-8 -*-

import tensorflow as tf
import os
import numpy as np

#-0.4076059881573251
print(np.log(0.66524094))

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logits = tf.constant([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype = tf.float32)
y = tf.nn.softmax(logits)
y_ = tf.constant([[0, 0, 1],[0, 0, 1],[0, 0, 1]], dtype = tf.float32)  
y__ = tf.constant([2, 2, 2], dtype = tf.int32)

# [[0.09003057 0.24472848 0.66524094]
#  [0.09003057 0.24472848 0.66524094]
#  [0.09003057 0.24472848 0.66524094]]
# 不用加负号
cross_value1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_))
cross_value2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y__))

with tf.Session() as sess:
    print(sess.run(y))

    #1.2228179
    #1.2228179
    print(sess.run(cross_value1))
    print(sess.run(cross_value2))
