# -*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype = tf.int32)

#[[1 2 3]
# [0 5 6]
# [0 0 9]]
# 上对角矩阵
b = tf.matrix_band_part(a, 0, -1)

#[[1 0 0]
# [4 5 0]
# [7 8 9]]
# 下对角矩阵
c = tf.matrix_band_part(a, -1, 0)

# diagonal, 对角矩阵
#[[1 0 0]
# [0 5 0]
# [0 0 9]]
d = tf.matrix_band_part(a, 0, 0)

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
