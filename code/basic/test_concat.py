# -*- coding:utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# (2, 3)
a = tf.constant([[1, 2, 3], [4, 5, 6]])
# (2, 3)
b = tf.constant([[7, 8, 9], [10, 11, 12]])

#axis=0，表示在第一维连接
#(4, 3) --> (2 * 2, 3)
concat_0 = tf.concat([a, b], axis = 0)
print(concat_0.shape)

#(2, 6) --> (2, 3 * 2)
concat_1 = tf.concat([a, b], axis = 1)
print(concat_1.shape)

#[1 2 3 4]
c = tf.concat([[1, 2], [3, 4]], axis = 0)

# Shape must be at least rank 2 but is rank 1 for 'concat_3'
# 会报错, 因为是按照里面 [1, 2] 的维度来计算的, 里面的维度是(2,), 并没有第二个维度
# d = tf.concat([[1, 2], [3, 4]], axis = 1)

with tf.Session() as sess:
    #[[ 1  2  3]
    # [ 4  5  6]
    # [ 7  8  9]
    # [10 11 12]]
    print(sess.run(concat_0))

    # (2, 6)
    #[[ 1  2  3  7  8  9]
    # [ 4  5  6 10 11 12]]
    print(sess.run(concat_1))

    print(sess.run(c))
