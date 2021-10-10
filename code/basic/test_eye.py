

# 生成单位矩阵

import tensorflow as tf


a = tf.eye(3, 3)
b = tf.eye(3)
c = tf.eye(2, 3)


with tf.Session() as sess:
   print(sess.run(a))
   print(sess.run(b))
   print(sess.run(c))


# result
#[[1. 0. 0.]
# [0. 1. 0.]
# [0. 0. 1.]]
#
#[[1. 0. 0.]
# [0. 1. 0.]
# [0. 0. 1.]]
#
# 非方阵情况下如何产出
#[[1. 0. 0.]
# [0. 1. 0.]]
