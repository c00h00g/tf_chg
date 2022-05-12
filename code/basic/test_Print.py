# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

a = np.random.randint(0, 10, size=(3, 5))
a_t = tf.convert_to_tensor(a)
a_t = tf.Print(a_t, [a_t], message="chg debug", first_n=2, summarize=3)

with tf.Session() as  sess:
    print(sess.run(a_t))

"""
chg debug[[0 8 2...]...]
[[0 8 2 5 4]
 [9 5 3 9 9]
 [2 4 6 5 7]]
"""
