
# 将numpy变量转换为tensor
# tf.where返回是坐标, tf.where()返回一个布尔张量中真值的位置。对于非布尔型张量，非0的元素都判为True

import tensorflow as tf
import numpy as np


a = np.random.randint(3, size = (4, 4))
a_t = tf.convert_to_tensor(a)
b = tf.where(a_t)


#[[1 2 2 1]
# [1 0 1 2]
# [2 2 1 2]
# [0 1 0 1]]
#
#[[0 0]
# [0 1]
# [0 2]
# [0 3]
# [1 0]
# [1 2]
# [1 3]
# [2 0]
# [2 1]
# [2 2]
# [2 3]
# [3 1]
# [3 3]]

with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))
