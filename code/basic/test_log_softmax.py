# -*- coding:utf-8 -*-
# tf.nn.log_soft的用法主要是在计算交叉熵的时候，-y * log(f(x)) - (1 - y) * log(1 - f(x))

import tensorflow as tf
import numpy as np

def calc_softmax(elem_list):
    sum_soft = 0.0
    for elem in elem_list:
        sum_soft += np.exp(elem)

    res = []
    for elem in elem_list:
        res.append(np.exp(elem) / sum_soft)
    return res

def calc_log_softmax(elem_list):
    sum_soft = 0.0
    for elem in elem_list:
        sum_soft += np.exp(elem)

    res = []
    for elem in elem_list:
        res.append(np.log(np.exp(elem) / sum_soft))
    return res

calc_softmax([1, 2, 3])
calc_log_softmax([1, 2, 3])

# [1. 2. 3.]
a = tf.constant([1, 2, 3], dtype = tf.float32)

# [0.09003057 0.24472848 0.66524094]
b = tf.nn.softmax(a)

# [-2.407606   -1.4076059  -0.40760595]
c = tf.nn.log_softmax(a)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
