# -*- coding:utf-8 -*-

import tensorflow as tf

# [0 1 2 3 4 5 6 7 8 9]
# 参数 
# start
# limit
# delta
a = tf.range(0, 10, delta = 1)

# [0 2 4 6 8]
b = tf.range(0, 10, delta = 2)

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)
