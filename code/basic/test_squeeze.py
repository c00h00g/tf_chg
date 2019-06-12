# -*- coding:utf-8 -*-

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#用法：从张量形状中移除大小为1的维度

a = tf.constant([[1], [2], [3]])
#(3,1)
print(a.shape)

#将1移除后，只剩下3
#(3,)
b = tf.squeeze(a)
print(b.shape)

c = tf.squeeze(a, [1])
#(3,)
print(c.shape)

with tf.Session() as tf:
    #[1 2 3]
    print(tf.run(b))

    #[1 2 3]
    print(tf.run(c))
