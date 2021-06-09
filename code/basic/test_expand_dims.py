# -*- coding:utf-8 -*-
import sys
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.constant([1, 2, 3], tf.int32)

#(1, 3)
b = tf.expand_dims(a, 0)

# (3, 1)
c = tf.expand_dims(a, [1]) 
c1 = tf.expand_dims(a, 1) 

with tf.Session() as sess:
    #[[1 2 3]]
    print(sess.run(b))

    #[[1]
    # [2]
    # [3]]
    print(sess.run(c))

    #[[1]
    # [2]
    # [3]]
    print(sess.run(c1))
