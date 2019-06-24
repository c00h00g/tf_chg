# -*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype = tf.int32)
# (2, 2, 3)
print(a.shape)

# (2, 2, 1, 3)
# 相当于扩充一个维度
b = a[:, :, None, :]
print(b.shape)

c = a[:, -1, :]
#(2, 3)
print(c.shape)

with tf.Session() as sess:
    #[[[ 1  2  3]
    #  [ 4  5  6]]

    # [[ 7  8  9]
    #  [10 11 12]]]
    print(sess.run(a))

    #[[[[ 1  2  3]]
    #  [[ 4  5  6]]]

    # [[[ 7  8  9]]
    # [[10 11 12]]]]
    print(sess.run(b))

    #[[ 4  5  6]
    # [10 11 12]]
    print(sess.run(c))
