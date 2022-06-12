# -*- coding:utf-8 -*-

"""
数据的排布为：
一行原样本
一行相似样本
a
a+
b
b+
"""

import keras.backend as K
import numpy as np
import tensorflow as tf

y_pred = np.random.randint(0, 5, size=(5, 8))
y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

y_shape = K.shape(y_pred)[0]
idxs = K.arange(0, y_shape)
idxs_1 = idxs[None,:]
idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]

y_true = K.equal(idxs_1, idxs_2)
y_true = K.cast(y_true, K.floatx())
y_pred = K.l2_normalize(y_pred, axis=1)

similarities = K.dot(y_pred, K.transpose(y_pred))
similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
similarities = similarities * 20
loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)


with tf.Session() as sess:
    print(sess.run(y_shape))
    print(sess.run(idxs))
    print(sess.run(idxs_1))
    print(sess.run(idxs_2))
    print("chg y_true is ------>")
    print(sess.run(y_true))
    print(sess.run(similarities))
