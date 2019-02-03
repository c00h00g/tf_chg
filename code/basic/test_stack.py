import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

#0按照行的方向stack
c = tf.stack([a, b], 0)

#1按照列的方向stack
d = tf.stack([a, b], 1)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))

#result c
#[[1 2 3]
# [4 5 6]]

#result d
#[[1 4]
# [2 5]
# [3 6]]
