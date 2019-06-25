# -*- coding:utf-8 -*-

"""
按照某个纬度拆分成数组
0 : 横着
1 : 竖着
"""

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 2 * 3 
a = tf.constant([[1, 2, 3], [4, 5, 6]])

# 按照第一个纬度拆分
b = tf.unstack(a, axis = 0)

c = b[0]
d = b[1]

# 按照第二个纬度拆分
e = tf.unstack(a, axis = 1)

with tf.Session() as sess:
    #[array([1, 2, 3], dtype=int32), array([4, 5, 6], dtype=int32)]
    print(sess.run(b))
    #[1 2 3]
    print(sess.run(c))
    #[4 5 6]
    print(sess.run(d))

    #[array([1, 4], dtype=int32), array([2, 5], dtype=int32), array([3, 6], dtype=int32)]
    print(sess.run(e))
    
