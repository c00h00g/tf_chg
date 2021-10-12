

import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = a[:, 0:2]
c = a[:, 2:]

#print(c)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

#outputs
#[[1 2 3]
# [4 5 6]]
#[[1 2]
# [4 5]]
#[[3]
# [6]]
