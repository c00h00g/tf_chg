# -*- coding:utf-8 -*-

import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

a = tf.constant([1, 2, 3], dtype = tf.int32)

# 三个参数
# (features, vocab_size, embed_dim)
output_1 = tf.contrib.layers.embed_sequence(a, 2, 5)

output_2 = tf.contrib.layers.embed_sequence(a, 10, 5)


# must be initialize
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #[[ 0.7947664   0.05015332 -0.25174564 -0.890464   -0.6450249 ]
    # [ 0.          0.          0.          0.          0.        ]
    # [ 0.          0.          0.          0.          0.        ]]
    # 索引一定要小于vocab_size, 大于等于vocab_size都会为0
    print(sess.run(output_1))

    #[[-0.36656886 -0.50490105 -0.4363452   0.5128432  -0.33388773]
    # [ 0.43284267 -0.29013914 -0.12848079 -0.1300888  -0.28769347]
    # [ 0.54997176  0.27302307 -0.2945259  -0.5524364  -0.01611257]]
    print(sess.run(output_2))
