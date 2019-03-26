import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[1, 1, 1], [1, 1, 1]])

#[[0 1 2]
# [3 4 5]]
c = tf.subtract(a, b)


with tf.Session() as sess:
    print(sess.run(c))
