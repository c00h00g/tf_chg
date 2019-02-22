import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


a = tf.constant([1, 2, 3])
b = tf.constant([1, 2, 3])

c = tf.constant([[1, 2, 3]])
d = tf.constant([[1], [2], [3]])

with tf.Session() as sess:
    #[1 4 9]
    print(sess.run(tf.multiply(a, b)))

    #[1 4 9]
    #矩阵点乘可以直接用乘号
    print(sess.run(a * b))

    #[[14]]
    print(sess.run(tf.matmul(c, d)))
