import tensorflow as tf

a = tf.one_hot(1, 4)
b = tf.one_hot(2, 4)
c = a + b

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

# output
#[0. 1. 0. 0.]
#[0. 0. 1. 0.]
#[0. 1. 1. 0.]
