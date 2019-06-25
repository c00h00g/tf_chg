# -*- coding:utf-8 -*-

"""
1. stack 会导致维度的增加
2. concat的维度不变
"""

import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[10, 20, 30], [40, 50, 60]])

# (2, 2, 3)
stack = tf.stack([a, b])
print(stack.shape)

# (2, 2, 3)
stack_0 = tf.stack([a, b], 0)
print(stack_0.shape)

# (2, 2, 3)
stack_1 = tf.stack([a, b], 1)
print(stack_1.shape)

# (4, 3)
concat_0 = tf.concat([a, b], 0)
print(concat_0.shape)

# (2, 6)
concat_1 = tf.concat([a, b], 1)
print(concat_1.shape)


with tf.Session() as sess:
    #[[[ 1  2  3]
    #  [ 4  5  6]]
    # [[10 20 30]
    #  [40 50 60]]]
    print(sess.run(stack))

    #[[[ 1  2  3]
    #  [ 4  5  6]]
    # [[10 20 30]
    #  [40 50 60]]]
    print(sess.run(stack_0))

    #[[[ 1  2  3]
    # [10 20 30]]
    # [[ 4  5  6]
    # [40 50 60]]]
    print(sess.run(stack_1))

    #[[ 1  2  3]
    # [ 4  5  6]
    # [10 20 30]
    # [40 50 60]]
    print(sess.run(concat_0))

    #[[ 1  2  3 10 20 30]
    #[ 4  5  6 40 50 60]]
    print(sess.run(concat_1))



