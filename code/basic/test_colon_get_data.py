import tensorflow as tf
import numpy as np


"""
c和d的区别是c会减少一个维度
"""

a = np.random.randint(0, 3, size=(5, 2))
a = tf.cast(a, dtype=tf.float32)
b = tf.nn.softmax(a)

c = b[:,1]
d = b[:, 0:1]

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))

"""
[[0. 0.]
 [1. 0.]
 [2. 1.]
 [2. 0.]
 [2. 0.]]
[[0.5        0.5       ]
 [0.7310586  0.26894143]
 [0.7310586  0.26894143]
 [0.880797   0.11920291]
 [0.880797   0.11920291]]
[0.5        0.26894143 0.26894143 0.11920291 0.11920291]
[[0.5      ]
 [0.7310586]
 [0.7310586]
 [0.880797 ]
 [0.880797 ]]
"""
