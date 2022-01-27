# -*- coding:utf-8 -*-

"""

"""


import tensorflow as tf


def test_1(sess):
    """
    [[[1 2 3]]

     [[4 5 6]]]

    [[1]
     [1]
     [1]]

    [[[ 6]]

     [[15]]]
     两个tensor的矩阵相乘
    """
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    # (2,1,3)
    a = tf.expand_dims(a, axis=1)

    b = tf.constant([1, 1, 1])
    #(3,1)
    b = tf.expand_dims(b, axis=1)

    # (2,1,1)
    c = tf.keras.backend.dot(a, b)

    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))



with tf.Session() as sess:
    print("chg test_1 is -------------->")
    test_1(sess)

