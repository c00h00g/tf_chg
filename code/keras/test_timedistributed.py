# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

a = np.random.randint(0, 5, size=(3, 4, 5))
a_t = tf.convert_to_tensor(a)
print('a_t shape is : %s' % (a_t.shape))

b = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(a_t)
print('b shape is : %s' % (b.shape))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

"""
a_t shape is : (3, 4, 5)
b shape is : (3, 4, 10)
相当于在改变最后一层的输出维度大小
"""
