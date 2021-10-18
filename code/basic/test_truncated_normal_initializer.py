# -*- coding:utf-8 -*-
# 解决梯度过小的问题，神经元会停止学习

import tensorflow as tf

a = tf.truncated_normal_initializer(stddev = 1)
b = tf.get_variable('b', shape = [3, 4], initializer = a)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
