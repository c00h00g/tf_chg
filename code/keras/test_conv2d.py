# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

a = np.random.randint(1, 2, size=(1, 3, 3, 4))
a_t = tf.convert_to_tensor(a, dtype=tf.float32)

initializer = tf.keras.initializers.Ones()
b = tf.keras.layers.Conv2D(5, 2, use_bias=False, kernel_initializer=initializer)(a_t)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print('a_t shape is : %s' % (a_t.shape))
    print(sess.run(a_t))

    print('b shape is : %s' % (b.shape))
    print(sess.run(b))

"""
a_t shape is : (1, 3, 3, 4)
[[[[1. 1. 1. 1.]
   [1. 1. 1. 1.]
   [1. 1. 1. 1.]]

  [[1. 1. 1. 1.]
   [1. 1. 1. 1.]
   [1. 1. 1. 1.]]

  [[1. 1. 1. 1.]
   [1. 1. 1. 1.]
   [1. 1. 1. 1.]]]]

b shape is : (1, 2, 2, 5)
[[[[16. 16. 16. 16. 16.]
   [16. 16. 16. 16. 16.]]

  [[16. 16. 16. 16. 16.]
   [16. 16. 16. 16. 16.]]]]

为什么是16？

3*3*4的图像经过2*2的filter之后，得到了
1 1 1
1 1 1  
1 1 1

经过2*2的filter后---->
4 4
4 4

因为是4个，所以结果是
16  16
16  16

因为有5个filter，所有最终是1 * 2 * 2 * 5
"""
