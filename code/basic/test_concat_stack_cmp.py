# -*- coding:utf-8 -*-
# 区别：stack会增加一个纬度，但是concat不会增加纬度

import tensorflow as tf

#(2, 3)
a = tf.constant([[1, 2, 3], [3, 4, 5]])
#(2, 3)
b = tf.constant([[7, 8, 9], [10, 11, 12]])

#(2 * 2, 3)
ab_1 = tf.concat([a, b], axis = 0)
#(2, 3 * 2)
ab_2 = tf.concat([a, b], axis = 1)

#(2, 2, 3)
ab_3 = tf.stack([a, b], axis = 0)
print(ab_3.shape)

#(2, 2, 3)
ab_4 = tf.stack([a, b], axis = 1)
print(ab_4.shape)

#(2, 3, 2)
ab_5 = tf.stack([a, b], axis = 2)
print(ab_5.shape)

with tf.Session() as sess:
    #[[ 1  2  3]
    # [ 3  4  5]
    # [ 7  8  9]
    # [10 11 12]]
    print(sess.run(ab_1))

    print('------------>')

    # [[ 1  2  3  7  8  9]
    #  [ 3  4  5 10 11 12]]
    print(sess.run(ab_2))

    print('------------>')

    #[[[ 1  2  3]
    #  [ 3  4  5]]

    #  [[ 7  8  9]
    #   [10 11 12]]]
    print(sess.run(ab_3))

    print('------------>')

    #[[[ 1  2  3]
    #  [ 7  8  9]]

    #  [[ 3  4  5]
    #   [10 11 12]]]
    print(sess.run(ab_4))

    print('------------>')

    #[[[ 1  7]
    #  [ 2  8]
    #  [ 3  9]]

    #  [[ 3 10]
    #   [ 4 11]
    #   [ 5 12]]]
    print(sess.run(ab_5))



