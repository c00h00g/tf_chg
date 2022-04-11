# -*- coding:utf-8 -*-
# weights格式如何使用的?

import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def np_func(x, label):
    x = sigmoid(x)
    if label == 1:
        return -np.log(x)
    else:
        return -np.log(1-x)


a = np.random.randint(0, 2, size=(2, 1))
a_t = tf.convert_to_tensor(a, dtype=tf.float32)

labels = np.random.randint(0, 2, size=(2, 1))
labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)

weights = np.random.randint(0, 2, size=(2, 1))
weights = tf.convert_to_tensor(weights, dtype=tf.float32)

sum_x = 0
for idx in range(a.shape[0]):
    x = a[idx]
    lb = labels[idx]
    print("x:%s, label:%s, loss:%s" % (x, lb, np_func(x, lb)))
    sum_x += -1.0 * np_func(x, lb)
print(sum_x/a.shape[0])


a_sig = tf.sigmoid(a_t)
res = tf.losses.log_loss(predictions=a_sig, labels=labels_t, weights=weights)


with tf.Session() as sess:
    print("chg a_t is ------>")
    print(sess.run(a_t))
    print(sess.run(a_sig))

    print("chg labels is ------>")
    print(sess.run(labels_t))

    print("chg weights is ------->")
    print(sess.run(weights))

    print("chg res is --->")
    print(sess.run(res))
