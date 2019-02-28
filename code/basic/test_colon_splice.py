# -*- coding:utf-8 -*-

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf


a = tf.constant([[[1, 2, 3], [4, 5, 6]],
                 [[7, 8, 9], [10, 11, 12]],
                 [[13, 14, 15], [16, 17, 18]]])
# (3 2 3)
print(a.shape)

b = a[:, 0:1, :]
c = a[:, :, 0:2]

with tf.Session() as sess:
    #[[[ 1  2  3]]
    # [[ 7  8  9]]
    # [[13 14 15]]]
    print(sess.run(b))
    #[[[ 1  2]
    #  [ 4  5]]
    #  [[ 7  8]
    #   [10 11]]
    #  [[13 14]
    #   [16 17]]]
    print(sess.run(c))
