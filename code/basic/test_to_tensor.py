# -*- coding:utf-8 -*-

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


a = [1, 2, 3]
b = tf.convert_to_tensor(a, dtype = tf.float32)
c = 5

print(a)
print(b)

with tf.Session() as sess:
    #[1. 4. 9.]
    print(sess.run(a * b))
    #[ 5. 10. 15.]
    print(sess.run(c * b))
