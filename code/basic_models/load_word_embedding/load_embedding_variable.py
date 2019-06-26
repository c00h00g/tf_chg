# -*- coding:utf-8 -*-

"""
要说明的问题是如何加载已经训练词向量到内存中
1) 可以fine-tune, 使用varibale
"""

import tensorflow as tf
import numpy as np


def load_emb(path):
    emb_mat = []
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip()
            line_sp = line.split(' ')
            emb_mat.append([float(x) for x in line_sp])
    return np.array(emb_mat)


w = tf.Variable(tf.zeros([5, 5]), dtype = tf.float32)
emb = tf.placeholder(tf.float32, [5, 5])
emb_init = w.assign(emb)

init = tf.global_variables_initializer()
emb_a = load_emb('./embedding')

with tf.Session() as sess:
    sess.run(init)
    #[[0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]]
    print(sess.run(w))

    sess.run(emb_init, feed_dict = {emb : emb_a})
    #[[-0.00932655  0.23722178  0.7495388   0.31212994  1.1572491 ]
    # [-0.47935936 -1.1780828  -0.14345106 -1.3364223   0.6504877 ]
    # [-1.7168014   0.77895546  0.2147284  -0.810833   -1.3102798 ]
    # [ 0.8650782  -1.0621126   0.01807257 -0.7221659  -1.6468552 ]
    # [-0.31872445  1.8378029  -0.91336685 -0.4353269  -0.7694902 ]]
    print(sess.run(w))
    # <class 'tensorflow.python.ops.variables.RefVariable'>
    print(type(w))
