# -*- coding:utf-8 -*-

"""
被mask的元素直接删除
"""

import tensorflow as tf

a = tf.constant([[2, 3], [4, 5]], dtype = tf.float32)
b = tf.constant([True, False], dtype=tf.bool)

c = tf.boolean_mask(a, b, axis=0)
d = tf.boolean_mask(a, b, axis=1)

"""
[[2. 3.]]

[[2.]
 [4.]]
"""

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))

