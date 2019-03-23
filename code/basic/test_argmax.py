# -*- coding: utf-8 -*-

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


a = tf.constant([0.1, 0.2, 0.4, 0.5])
b = tf.constant([0.1, 0.1, 0.1, 0.1])

c = tf.constant([[1, 2, 3], [4, 5, 6]], dtype = tf.int32)

with tf.Session() as sess:
    # 3 返回最大的索引
    print(sess.run(tf.argmax(a)))
    # 0 返回index最小的
    print(sess.run(tf.argmax(b)))
    
    # [1 1 1]
    print(sess.run(tf.argmax(c, axis = 0)))
    # [2 2]
    print(sess.run(tf.argmax(c, axis = 1)))
