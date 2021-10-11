
import tensorflow as tf

a = tf.range(0, 5, dtype = tf.int32)

with tf.Session() as sess:
    print(sess.run(a))


# output
# [0 1 2 3 4]
