# -*- coding:utf-8 -*-

"""
tf.clip_by_value(A, min, max)������һ������A����A�е�ÿһ��Ԫ�ص�ֵ��ѹ����min��max֮�䡣
С��min����������min������max��Ԫ�ص�ֵ����max��
"""

import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.constant([[1, 20, 5, 6], [20, 1, 2, 3]])

#[[2 5 5 5]
# [5 2 2 3]]
b = tf.clip_by_value(a, 2, 5)

with tf.Session() as sess:
    print(sess.run(b))
