

import tensorflow as tf
import numpy as np

a = np.random.randint(0, 2, size=(2, 3, 4))
a_t = tf.convert_to_tensor(a)
a_t = tf.cast(a_t, tf.float32)

# (2, 1, 4)
b = tf.keras.backend.sum(a_t, axis=1, keepdims=True)
print('chg b is ---------->')
print(b)

c = tf.keras.backend.sum(a_t, axis=1, keepdims=False)

with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))
    print(sess.run(c))


""
a:"
[[[1. 1. 0. 1.]
  [1. 0. 0. 0.]
  [1. 0. 0. 1.]]

 [[0. 0. 1. 1.]
  [1. 1. 0. 1.]
  [0. 1. 0. 0.]]]

b:
[[[3. 1. 0. 2.]]
 [[1. 2. 1. 2.]]]

c: 会少一个维度
[[3. 1. 0. 2.]
 [1. 2. 1. 2.]]
"""
