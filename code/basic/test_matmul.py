# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

a = np.random.rand(3, 4, 5)
a_t = tf.convert_to_tensor(a, dtype=tf.float32)

b = np.random.rand(4, 5)
b_t = tf.convert_to_tensor(b, dtype=tf.float32)


# ValueError: Dimensions must be equal, but are 5 and 4 for 'MatMul' (op: 'BatchMatMulV2') with input shapes: [3,4,5], [5,4].
# c = tf.matmul(a, b, transpose_b = True)
# 由于a和b维度不同，使用matmul会报错，可以通过reshap处理
a_2 = tf.reshape(a_t, [-1, 5])
c_2 = tf.matmul(a_2, b_t, transpose_b = True)
c_3 = tf.reshape(c_2, [-1, 4, 4])

#(12, 5)
#(12, 4)
#(3, 4, 4)
print("a_2 shape is --->")
print(a_2.shape)
print(c_2.shape)
print(c_3.shape)

with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b_t))
    print(sess.run(a_2))
    print(sess.run(c_2))
    print(sess.run(c_3))
