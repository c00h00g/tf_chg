import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]])
b = 1 - a

#[[1 2]
# [3 4]]

#[[ 0 -1]
# [-2 -3]]

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))


