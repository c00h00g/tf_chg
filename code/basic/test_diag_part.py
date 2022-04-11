# -*- coding:utf-8 -*-
"""
返回tensor的对角线部分
一定要是方阵
"""

import tensorflow as tf
import numpy as np

a = np.random.randint(0, 10, size=(5, 5))
a_t = tf.convert_to_tensor(a)

b = tf.diag_part(a_t)



with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))

"""
[[6 4 3 4 2]
 [1 2 9 5 8]
 [6 0 5 0 3]
 [1 4 7 6 7]
 [7 4 7 6 2]]

[6 2 5 6 2]
"""
