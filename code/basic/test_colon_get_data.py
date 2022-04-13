import tensorflow as tf
import numpy as np

a = np.random.randint(0, 3, size=(5, 2))
a = tf.cast(a, dtype=tf.float32)
b = tf.nn.softmax(a)

c = b[:,1]


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

"""
[[0. 0.]
 [0. 1.]
 [1. 0.]
 [1. 2.]
 [2. 1.]]
[[0.5        0.5       ]
 [0.26894143 0.7310586 ]
 [0.7310586  0.26894143]
 [0.26894143 0.7310586 ]
 [0.7310586  0.26894143]]
# 会少一个维度
[0.5        0.7310586  0.26894143 0.7310586  0.26894143]
"""
