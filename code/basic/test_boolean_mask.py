
import tensorflow as tf

a = tf.constant([1, 2, 3], dtype = tf.float32)
b = tf.constant([True, False, True], dtype=tf.bool)

c = tf.boolean_mask(a, b)


with tf.Session() as sess:
    print(sess.run(c))


"""
[1. 3.]
"""
