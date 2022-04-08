# -*- coding:utf-8 -*-

import tensorflow as tf

logits = tf.constant([[1, 2], [3, 4]], dtype = tf.float32)
labels = tf.constant([0, 1], dtype = tf.int32)
weights = tf.constant([1, 2], dtype=tf.float32)


logits_softmax = tf.nn.softmax(logits)
labels_new = tf.one_hot(labels, depth=2, dtype=tf.float32)
loss1 = -tf.reduce_mean(tf.reduce_sum(tf.log(logits_softmax) * labels_new, axis=-1))


# loss2会自动做softmax操作
loss2 = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)


with tf.Session() as sess:
    print(sess.run(labels_new))
    print(sess.run(loss1))
    print(sess.run(loss2))

# outputs
#1.4076059
#1.4076059
