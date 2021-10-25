# -*- coding:utf-8 -*-


import tensorflow as tf

x = tf.placeholder(tf.int32, shape=[None, 3])
y = tf.placeholder(tf.int32, shape=[None, 3])
mask = tf.placeholder(tf.int32, shape=[None, 3])

acc, acc_op = tf.metrics.accuracy(labels=x, predictions=y, weights=mask)


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print(sess.run([acc_op], feed_dict={x:[[1, 2, 3], [4, 5, 6]], y:[[1, 2, 3], [4, 5, 5]], mask:[[1, 1, 1], [1, 1, 1]]}))
    print(sess.run([acc_op], feed_dict = {x:[[1, 2, 3], [4, 5, 6]], y:[[1, 2, 3], [4, 5, 5]], mask:[[1, 1, 0], [0, 0, 1]]}))

#outputs:
# [0.8333333] = 5/6
# [0.7777778] = 2/3
