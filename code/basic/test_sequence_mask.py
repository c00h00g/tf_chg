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
    # 第一个1表示mask第一个
    # 第二个2表示mask前两个
    # 第三个3表示mask前三个
    # 第四个4表示mask全部
    # 第二个参数表示总长度
    print(sess.run(seq_mask))

    # 全部mask
    # [ True  True  True  True  True  True  True  True  True  True]
    print(sess.run(a))

    # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    print(sess.run(b))
