# -*- coding:utf-8 -*-

import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
output = tf.contrib.layers.embed_sequence(a, 10, 6)

with tf.Session() as sess:
    print(sess.run(output))
