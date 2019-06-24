import sys
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf

t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])

#Tensor("Const:0", shape=(3, 2, 3), dtype=int32)
print(t)

#Tensor("Slice:0", shape=(1, 2, 3), dtype=int32)
s_res1 = tf.slice(t, [1, 0, 0], [1, 2, 3])
s_res2 = tf.slice(t, [1, 0, 0], [2, 2, 2])
s_res3 = tf.slice(t, [1, 0, 0], [1, 2, -1])

with tf.Session() as sess:
    #shape: (1, 2, 3)
    #[[[3 3 3]
    #  [4 4 4]]]
    print(s_res1.shape)
    print(sess.run(s_res1))

    #shape: (2, 2, 2)
    #[[[3 3]
    #  [4 4]]
    # [[5 5]
    #  [6 6]]]
    print(s_res2.shape)
    print(sess.run(s_res2))

    #[[[3 3 3]
    #  [4 4 4]]]
    print(sess.run(s_res3))

    # False
    print(s_res1 == s_res3)

    #[[[ True  True  True]
    #  [ True  True  True]]]
    print(sess.run(tf.equal(s_res1, s_res3)))
