# -*- coding:utf-8 -*-

"""
主要用来对文本进行处理, 一段话是一个序列
每个词是一个embedding，可以在窗口上进行卷积操作
"""

import tensorflow as tf
import numpy as np



a = np.random.randint(1, 2, size=(1, 8, 5))
a_t = tf.convert_to_tensor(a, dtype=tf.float32)

initializer = tf.keras.initializers.Ones()
b = tf.keras.layers.Conv1D(2, 3, use_bias=False, kernel_initializer=initializer)(a_t)

c = tf.keras.layers.Conv1D(2, 3)(a_t)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('a shape is : %s' % (a_t.shape))
    print(sess.run(a_t))

    print('b shape is : %s' % (b.shape))
    print(sess.run(b))

    print('c shape is : %s' % (c.shape))
    print(sess.run(c))


"""
a shape is : (1, 8, 5)
[[[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]]
b shape is : (1, 6, 2)
[[[15. 15.]
  [15. 15.]
  [15. 15.]
  [15. 15.]
  [15. 15.]
  [15. 15.]]]

c shape is : (1, 6, 2)
其中一列为一个channel
[[[-0.21098739 -1.3480828 ]
  [-0.21098739 -1.3480828 ]
  [-0.21098739 -1.3480828 ]
  [-0.21098739 -1.3480828 ]
  [-0.21098739 -1.3480828 ]
  [-0.21098739 -1.3480828 ]]]

"""



