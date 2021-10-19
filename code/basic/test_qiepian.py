# 测试切片操作，冒号的作用
# 注意如果切片的时候只写一个数字，维度会减少

import tensorflow as tf
import numpy as np


a = np.random.rand(3, 2)
a_t = tf.convert_to_tensor(a)

b = a_t[:, 1]
b_1 = a_t[:, 1:2]
c = tf.reshape(b, [-1, 1])


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))
    print(sess.run(b_1))
    print(sess.run(c))

# outputs
#[[0.60390435 0.61573024]
# [0.56551862 0.98427985]
# [0.69861979 0.76528663]]
#
#[0.61573024 0.98427985 0.76528663]
#
#[[0.61573024]
# [0.98427985]
# [0.76528663]]
#
#[[0.61573024]
# [0.98427985]
# [0.76528663]]
