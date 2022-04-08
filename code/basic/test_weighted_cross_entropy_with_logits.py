# -*- coding:utf-8 -*-

# label不需要进行onehot
# 只能处理二分类的问题

import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def np_func(x, label):
    a = sigmoid(x)

    if label == 1:
        return -np.log(a)
    else:
        return -np.log(1-a)

print(np_func(1, 1))


a = np.random.randint(0, 2, size=(3, 1))
a_t = tf.convert_to_tensor(a, dtype=tf.float32)
print(a_t)

b = tf.constant([[1], [0], [1]], dtype=tf.float32)

c = tf.nn.weighted_cross_entropy_with_logits(logits=a_t, targets=b, pos_weight=1.0)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("chg a_t is ------------>")
    print(sess.run(a_t))

    print("chg b is ------------>")
    print(sess.run(b))

    print("chg c is ---------->")
    print(sess.run(c))

