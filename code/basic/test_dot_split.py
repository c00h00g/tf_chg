import tensorflow as tf
import numpy as np

# 会减少一个维度

a = np.random.randint(0, 5, size=(2, 3, 4))
a_t = tf.convert_to_tensor(a)
b = [a_t[:,:,i] for i in range(4)]


with tf.Session() as sess:
    print(sess.run(a_t))
    print('-------------->')
    for elem in b:
        print(sess.run(elem))
        print('---------->')

"""
[[[0 1 0 4]
  [0 1 3 0]
  [3 3 4 1]]

 [[0 0 2 4]
  [2 4 1 3]
  [2 4 0 3]]]
-------------->
[[0 0 3]
 [0 2 2]]
---------->
[[1 1 3]
 [0 4 4]]
---------->
[[0 3 4]
 [2 1 0]]
---------->
[[4 0 1]
 [4 3 3]]
---------->
"""
