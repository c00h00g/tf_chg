# -*- coding:utf-8 -*-

#tf.where(
#    condition,
#    x=None,
#    y=None,
#    name=None
#)
# Return the elements, either from x or y, depending on the condition

import tensorflow as tf
import numpy as np


a = np.random.randint(0, 2, size=(2, 3))
cond = tf.convert_to_tensor(a)
cond = tf.cast(cond, tf.bool)

b = np.random.randint(1, 5, size=(2, 3))
b_t = tf.convert_to_tensor(b)

c = np.random.randint(6, 10, size=(2, 3))
c_t = tf.convert_to_tensor(c)

d = tf.where(cond, b_t, c_t)

with tf.Session() as sess:
    print('chg cond is -------->')
    print(sess.run(cond))

    print('chg b_t is -------->')
    print(sess.run(b_t))

    print('chg c_t is -------->')
    print(sess.run(c_t))

    print('chg d is -------->')
    print(sess.run(d))


"""
chg cond is -------->
[[False  True  True]
 [False False  True]]
chg b_t is -------->
[[3 2 4]
 [1 3 1]]
chg c_t is -------->
[[9 6 6]
 [7 8 9]]
chg d is -------->
[[9 2 4]
 [7 8 1]]
cond=True的时候取b_t的值, cond=False的时候取c_t的值
"""



