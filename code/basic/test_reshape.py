# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

"""
一个参数为-1时，那么reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值
"""

a = np.array([[1, 2, 3], [4, 5, 6]])  

with tf.Session() as sess:
    # 表示只有一个维度, -1表示如果一维的话，可以推断出来有多少个数
    #[1 2 3 4 5 6] 一维数组
    print(sess.run(tf.reshape(a, [-1])))

    #[[1 2 3 4 5 6]] 二维数组
    print(sess.run(tf.reshape(a, [1, -1])))

    #[[1 2]
    # [3 4]
    # [5 6]]
    print(sess.run(tf.reshape(a, [3, 2])))
    print(sess.run(tf.reshape(a, [3, -1])))

