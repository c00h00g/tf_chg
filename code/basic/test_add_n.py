# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np

# 输入: list


a = tf.convert_to_tensor(np.array([[1, 2, 3], [2, 3, 4]]), dtype = tf.float32)
b = tf.unstack(a)
c = tf.add_n(b)


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

# outputs:
#[[1. 2. 3.]
# [2. 3. 4.]]

#[array([1., 2., 3.], dtype=float32), array([2., 3., 4.], dtype=float32)]

#[3. 5. 7.]
