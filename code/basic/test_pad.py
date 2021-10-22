# -*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])

pad = [[0, 0], # 第一位维度左边:0, 右边：0
       [0, 5]  # 第二个维度，左边：0，右边：5
      ]

b = tf.pad(a, pad)


with tf.Session() as sess:
    print(sess.run(b))

# 第一个维度不做pad
# 第二个维度，右边做5次pad
# outputs:
#[[1 2 3 0 0 0 0 0]
# [4 5 6 0 0 0 0 0]]
