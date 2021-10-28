# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


a = np.random.rand(3, 4)
a_t = tf.convert_to_tensor(a)

#ids:
#[
#    [nan, 0, nan, nan],
#    [nan, nan, nan, nan],
#    [nan, nan, nan, nan],
#    [nan, nan, nan, nan],
#]


ids = tf.SparseTensor(indices=[[0, 1]], values=[0], dense_shape=[4, 4])
b = tf.nn.embedding_lookup_sparse(a, ids, None)

ids_2 = tf.SparseTensor(indices=[[0, 1], [0, 2]], values=[0, 1], dense_shape=[4, 4])
c = tf.nn.embedding_lookup_sparse(a, ids_2, None)


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))
    print(sess.run(c))

# outputs
#[[0.16216123 0.8021672  0.30517426 0.74443574]
# [0.58338457 0.28563309 0.96798152 0.72798532]
# [0.91786307 0.62498017 0.96721149 0.00110875]]

# 取第0个元素
#[[0.16216123 0.8021672  0.30517426 0.74443574]]

# 0 和 1向量求平均值
#[[0.3727729  0.54390014 0.63657789 0.73621053]]
