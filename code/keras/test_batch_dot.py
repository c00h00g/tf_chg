
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


a = np.random.randint(0, 3, size=(3, 2))
a_t = tf.convert_to_tensor(a)

b = np.random.randint(0, 3, size=(3, 2))
b_t = tf.convert_to_tensor(b)

# a_t : (3, 2)
# b_t : (3, 2)
# output: 3 * 1
c = tf.keras.backend.batch_dot(a_t, b_t, axes=1)


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b_t))
    print(sess.run(c))
    
#ouptus
#[[0 0]
# [1 1]
# [0 1]]
#[[0 1]
# [1 2]
# [2 1]]
#[[0]
# [3]
# [1]]
