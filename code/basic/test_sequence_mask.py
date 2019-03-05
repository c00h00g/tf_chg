# -*- coding:utf-8 -*-

import sys
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

#Tensor("SequenceMask/Less:0", shape=(4, 4), dtype=bool)
seq_mask = tf.sequence_mask([1, 2, 3, 4], 4)
a = tf.sequence_mask(10)
b = tf.cast(a, tf.float32)

with tf.Session() as sess:
    #[[ True False False False]
    # [ True  True False False]
    # [ True  True  True False]
    # [ True  True  True  True]]
    # seq_mask shape是 (4, 4)
    # 第一个1表示mask从1开始的所有
    # 第二个2表示mask从2开始的所有
    # 第三个3表示mask从3开始的所有
    # 第四个4表示mask从4开始的所有，但是已经没有了
    print(sess.run(seq_mask))

    # 一个都不mask
    # [ True  True  True  True  True  True  True  True  True  True]
    print(sess.run(a))

    # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    print(sess.run(b))
