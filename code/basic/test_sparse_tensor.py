# -*- coding:utf-8 -*-

import tensorflow as tf
a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
b = tf.sparse_tensor_to_dense(a)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))

#outputs
# indices表示坐标，values表示坐标对应的值，dense_shape表示真实的维度
#SparseTensorValue(indices=array([[0, 0],
#       [1, 2]]), values=array([1, 2], dtype=int32), dense_shape=array([3, 4]))
#
#[[1 0 0 0]
# [0 0 2 0]
# [0 0 0 0]]
