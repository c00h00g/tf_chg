# -*- coding:utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[7, 8, 9], [10, 11, 12]])

#axis=0，表示在第一维连接
concat_0 = tf.concat([a, b], axis = 0)

concat_1 = tf.concat([a, b], axis = 1)

with tf.Session() as sess:
    #[[ 1  2  3]
    # [ 4  5  6]
    # [ 7  8  9]
    # [10 11 12]]
    print(sess.run(concat_0))

    #[[ 1  2  3  7  8  9]
    # [ 4  5  6 10 11 12]]
    print(sess.run(concat_1))
