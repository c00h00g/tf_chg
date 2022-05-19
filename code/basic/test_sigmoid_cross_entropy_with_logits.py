# -*- coding:utf-8 -*-

"""
logits是一维的情况
"""

import tensorflow as tf
import numpy as np


a = np.random.randint(0, 5, size=(3, 6))
a_t = tf.convert_to_tensor(a, dtype=tf.float32)


labels = np.random.randint(0, 2, size=(3, 6))
labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_t, logits=a_t)


### 另一种方法
sig_a = tf.sigmoid(a_t)
cond = tf.cast(labels, dtype=tf.bool)
res = tf.where(cond, -tf.log(sig_a), -tf.log(1-sig_a))

with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(labels_t))
    print(sess.run(loss))
    print("sig_a is ------->>")
    print(sess.run(sig_a))
    print(sess.run(cond))
    print(sess.run(res))


"""
[[2. 1. 2. 2. 0. 3.]
 [4. 4. 0. 3. 3. 0.]
 [3. 3. 1. 3. 2. 1.]]
[[0. 1. 0. 1. 1. 0.]
 [0. 0. 1. 0. 1. 0.]
 [1. 0. 0. 0. 0. 1.]]
[[2.126928   0.3132617  2.126928   0.12692802 0.6931472  3.0485873 ]
 [4.01815    4.01815    0.6931472  3.0485873  0.04858735 0.6931472 ]
 [0.04858735 3.0485873  1.3132617  3.0485873  2.126928   0.3132617 ]]
sig_a is ------->>
[[0.8807971  0.7310586  0.8807971  0.8807971  0.5        0.95257413]
 [0.98201376 0.98201376 0.5        0.95257413 0.95257413 0.5       ]
 [0.95257413 0.95257413 0.7310586  0.95257413 0.880797   0.7310586 ]]
[[False  True False  True  True False]
 [False False  True False  True False]
 [ True False False False False  True]]
[[2.126928   0.31326166 2.126928   0.126928   0.6931472  3.0485876 ]
 [4.0181484  4.0181484  0.6931472  3.0485876  0.04858734 0.6931472 ]
 [0.04858734 3.0485876  1.3132617  3.0485876  2.1269276  0.31326166]]
"""
