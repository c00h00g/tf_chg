# -*- coding:utf-8 -*-

"""
tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
小于min的让它等于min，大于max的元素的值等于max。
"""

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.constant([[1, 20, 5, 6], [20, 1, 2, 3]])

#[[2 5 5 5]
# [5 2 2 3]]
b = tf.clip_by_value(a, 2, 5)

with tf.Session() as sess:
    print(sess.run(b))
