# -*- coding:utf-8 -*-

#比如，输入为:
#[[1, 2, 3],
# [4, 5, 6],
# [7, 8, 9]
#]
#输出为:
#[
#  [[1, 2, 3], 
#   [1, 2, 3]]
#  [[4, 5, 6], 
#   [4, 5, 6]]
#  [[7, 8, 9], 
#   [7, 8, 9]]
#]

import tensorflow as tf
import numpy as np


a = np.random.randint(0, 10, size = (3, 3))
a_t = tf.convert_to_tensor(a)
a_shape = a_t.shape
print("chg a_shape is ----------->")
print(a_shape)

b = tf.tile(a_t, [1, 3])
c = tf.reshape(b, [-1, 3, a_shape[-1]])


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))
    print(sess.run(c))

#outputs:
#[[4 5 8]
# [4 8 5]
# [2 5 4]]
#[[4 5 8 4 5 8 4 5 8]
# [4 8 5 4 8 5 4 8 5]
# [2 5 4 2 5 4 2 5 4]]
#[[[4 5 8]
#  [4 5 8]
#  [4 5 8]]
#
# [[4 8 5]
#  [4 8 5]
#  [4 8 5]]
#
# [[2 5 4]
#  [2 5 4]
#  [2 5 4]]]
