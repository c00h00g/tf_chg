
import tensorflow as tf

a = tf.range(0, 5, dtype = tf.int32)
b = a * 3
c = tf.reshape(b, [-1, 1])

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))


# output
# [0 1 2 3 4]
# [ 0  3  6  9 12]
# [[ 0]
#  [ 3]
#  [ 6]
#  [ 9]
#  [12]]
