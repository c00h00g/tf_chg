# -*- coding:utf-8 -*-

# 截断后的正太分布

import tensorflow as tf

def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)
