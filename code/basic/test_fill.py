# -*- coding:utf-8 -*-
"""
1. ��valueΪ0ʱ����ͬ��tf.zeros
2. ��valueΪ1ʱ����ͬ��tf.ones
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
