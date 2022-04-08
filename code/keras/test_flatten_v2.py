
"""
Flattens the input. Does not affect the batch size.
在非batch层展开
"""
import tensorflow as tf
import numpy as np

a = np.random.randint(0, 10, size=(3, 1, 5))
a_t = tf.convert_to_tensor(a)


b = tf.keras.layers.Flatten()(a_t)


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))


"""
[[[7 0 6 9 0]
  [8 7 4 5 8]
  [4 6 6 5 9]
  [7 1 8 1 4]]

 [[5 0 0 5 9]
  [2 2 7 8 5]
  [7 1 6 2 3]
  [0 8 3 8 9]]

 [[7 3 0 4 9]
  [3 8 9 9 2]
  [7 2 5 6 3]
  [1 3 1 3 4]]]
[[7 0 6 9 0 8 7 4 5 8 4 6 6 5 9 7 1 8 1 4]
 [5 0 0 5 9 2 2 7 8 5 7 1 6 2 3 0 8 3 8 9]
 [7 3 0 4 9 3 8 9 9 2 7 2 5 6 3 1 3 1 3 4]]
 """
