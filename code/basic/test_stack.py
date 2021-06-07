# -*- coding:utf-8 -*-

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

#0按照行的方向stack
c = tf.stack([a, b], 0)

#1按照列的方向stack
d = tf.stack([a, b], 1)

# [a, b]是一个列表
e = tf.stack([a, b])

#with tf.Session() as sess:
#with tf.compat.v1.Session() as sess:

sess= tf.compat.v1.Session()
print(sess.run(c))
print(sess.run(d))

# 默认按照第一个维度
#[[1 2 3]
# [4 5 6]]
print(sess.run(e))

#result c
#[[1 2 3]
# [4 5 6]]

#result d
#[[1 4]
# [2 5]
# [3 6]]
