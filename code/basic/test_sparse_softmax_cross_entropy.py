# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

num_classes = 3
batch_size = 3

l = np.array([1, 0, 2])
label_1 = tf.convert_to_tensor(l, dtype = tf.int32)

l_2 = np.array([[0, 1, 0],[1, 0, 0], [0, 0, 1]])
label_2 = tf.convert_to_tensor(l_2, dtype = tf.int32)

# 3 * 3
logits = tf.convert_to_tensor(np.array([[0.1, 2, 3],[1, 2, 0], [0.4, 0, 1]]), dtype = tf.float32)

loss_1 = tf.losses.sparse_softmax_cross_entropy(labels = label_1, logits = logits)
loss_2 = tf.losses.softmax_cross_entropy(onehot_labels = label_2, logits = logits)

init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run([init, init_local])
    np.set_printoptions(precision=4, suppress=True)

    l1 = sess.run(loss_1)
    print("tf.losses.sparse_softmax_cross_entropy() loss")
    print(l1)

    l2 = sess.run(loss_2)
    print("tf.losses.softmax_cross_entropy() loss")
    print(l2)


## outputs
#tf.losses.sparse_softmax_cross_entropy() loss
#1.1369685
#tf.losses.softmax_cross_entropy() loss
#1.1369685
