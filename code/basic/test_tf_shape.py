# -*- coding:utf-8 -*-

import sys
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


#(3, 2, 3)
a = tf.constant([[[1, 2, 3], [4, 5, 6]],
                 [[7, 8, 9], [10, 11, 12]],
                 [[11, 12, 13], [14, 15, 16]]])

with tf.Session() as sess:
    #[3 2 3]
    print(sess.run(tf.shape(a)))

    #[3 2]
    print(sess.run(tf.shape(a)[:-1]))
