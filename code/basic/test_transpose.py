# -*- coding:utf-8 -*-

import tensorflow as tf



x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype = tf.int32)
x_1 = tf.transpose(x, [1, 0])

y = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype = tf.int32)
y_1 = tf.transpose(y, [0, 2, 1])
y_2 = tf.transpose(y, [1, 0, 2])


#[[1 2 3]
# [4 5 6]]

#[[1 4]
# [2 5]
# [3 6]]

# ==============>
#[[[ 1  2  3]
#  [ 4  5  6]]
#
# [[ 7  8  9]
#  [10 11 12]]]
#
#[[[ 1  4]
#  [ 2  5]
#  [ 3  6]]
#
# [[ 7 10]
#  [ 8 11]
#  [ 9 12]]]
#
#[[[ 1  2  3]
#  [ 7  8  9]]
#
# [[ 4  5  6]
#  [10 11 12]]]

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(x_1))

    print(sess.run(y))
    print(sess.run(y_1))
    print(sess.run(y_2))
