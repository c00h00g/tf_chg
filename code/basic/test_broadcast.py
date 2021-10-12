# -*- coding:utf-8 -*-

"""
"""

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype = tf.float32)
print(a)

b = tf.constant([[[0.5, 0.5]], [[0.6, 0.6]]], dtype = tf.float32)
print(b)

c = a * b
print(c)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

# output
# 2 * 2 * 2
#[[[1. 1.]
#  [2. 2.]]
#
# [[3. 3.]
#  [4. 4.]]]
#
# 2 * 1 * 2 ---> 2 * 2 * 2
#[[[0.5 0.5]]
#
# [[0.6 0.6]]]
#[[[0.5       0.5      ]
#  [1.        1.       ]]
#
# [[1.8000001 1.8000001]
#  [2.4       2.4      ]]]
    
