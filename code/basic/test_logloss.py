# -*- coding:utf-8 -*-

import tensorflow as tf

logits = tf.constant([1, 2, 3], dtype = tf.float32)
labels = [0, 1, 0]


logits_softmax = tf.nn.softmax(logits)
loss1 = -tf.reduce_sum(tf.log(logits_softmax) * labels)


# loss2会自动做softmax操作
loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)


with tf.Session() as sess:
    print(sess.run(loss1))
    print(sess.run(loss2))
