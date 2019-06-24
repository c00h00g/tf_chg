# -*- coding:utf-8 -*-
"""
1. 当value为0时，等同于tf.zeros
2. 当value为1时，等同于tf.ones
"""

import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.fill([3, 4], 6)

with tf.Session() as sess:
    #[[6 6 6 6]
    #[6 6 6 6]
    #[6 6 6 6]]
    print(sess.run(a))
