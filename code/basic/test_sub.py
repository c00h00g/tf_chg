import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[1, 1, 1], [1, 1, 1]])

c = tf.subtract(a, b)
d = tf.subtract(a, 3)


with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))

# outputs
#[[0 1 2]
# [3 4 5]]
#[[-2 -1  0]
# [ 1  2  3]]
