# -*- coding:utf-8 -*-
# round四舍五入取整

#[[0.34438072]
# [0.27261582]
# [0.60359005]
# [0.33297715]
# [0.13614935]]
#[[0.]
# [0.]
# [1.]
# [0.]
# [0.]]

import tensorflow as tf
import numpy as np

a = np.random.rand(5, 1)
a_t = tf.convert_to_tensor(a)
b = tf.round(a_t)


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))
