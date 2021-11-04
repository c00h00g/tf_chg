# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

# [1, 10)区间随机取，shape = (3, 4)
a = np.random.randint(1, 10, size = (3, 4))
a_t = tf.convert_to_tensor(a)

b = tf.argsort(a_t, axis = -1)
c = tf.argsort(a_t, axis = -1, direction = 'DESCENDING')


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))
    print(sess.run(c))

# outputs
# 返回的是排序后的位置的索引
#
#[[6 7 5 2]
# [4 1 4 5]
# [1 3 4 9]]
#
# 正序
#[[3 2 0 1]
# [1 0 2 3]
# [0 1 2 3]]
#
# 逆序
#[[1 0 2 3]
# [3 0 2 1]
# [3 2 1 0]]
