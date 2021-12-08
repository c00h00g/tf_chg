
import tensorflow as tf

a = tf.ones(shape=[3, 4], dtype=tf.int32)
b = tf.zeros(shape=[3, 4], dtype=tf.int32)


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))

#[[1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]]
#
#[[0 0 0 0]
# [0 0 0 0]
# [0 0 0 0]]
