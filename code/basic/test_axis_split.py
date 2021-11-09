# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np

a = np.random.randint(0, 10, size = (3, 4, 5))
a_t = tf.convert_to_tensor(a)

b = tf.split(a, [1, 2, 1], axis=1)


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))

#[[[4 6 6 9 0]
#  [7 5 3 5 8]
#  [3 7 7 9 9]
#  [0 6 0 2 6]]
#
# [[6 8 0 4 5]
#  [3 2 2 7 3]
#  [2 6 5 4 2]
#  [9 3 3 9 1]]
#
# [[1 3 4 9 3]
#  [4 0 1 9 4]
#  [9 4 7 2 6]
#  [9 0 0 5 4]]]
#[array([[[4, 6, 6, 9, 0]],
#
#       [[6, 8, 0, 4, 5]],
#
#       [[1, 3, 4, 9, 3]]]), array([[[7, 5, 3, 5, 8],
#        [3, 7, 7, 9, 9]],
#
#       [[3, 2, 2, 7, 3],
#        [2, 6, 5, 4, 2]],
#
#       [[4, 0, 1, 9, 4],
#        [9, 4, 7, 2, 6]]]), array([[[0, 6, 0, 2, 6]],
#
#       [[9, 3, 3, 9, 1]],
#
#       [[9, 0, 0, 5, 4]]])]
