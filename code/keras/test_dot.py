# -*- coding:utf-8 -*-


import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
# (2,1,3)
a = tf.expand_dims(a, axis=1)

b = tf.constant([1, 1, 1])
#(3,1)
b = tf.expand_dims(b, axis=1)

# (2,1,1)
c = tf.keras.backend.dot(a, b)


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))



"""
[[[1 2 3]]

 [[4 5 6]]]

[[1]
 [1]
 [1]]

[[[ 6]]

 [[15]]]
"""
