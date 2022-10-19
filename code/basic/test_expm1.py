import tensorflow as tf

a = tf.constant([0, 1, 2, 3], dtype=tf.float32)
b = tf.expm1(a)


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))

"""
e^x-1
[0. 1. 2. 3.]
[ 0.         1.7182819  6.389056  19.085537 ]
"""
