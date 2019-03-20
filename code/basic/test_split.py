# -*- coding:utf-8 -*-

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


#(3, 2, 3)
a = tf.constant([[[1, 2, 3], [4, 5, 6]],
                 [[7, 8, 9], [10, 11, 12]],
                 [[13, 14, 15], [16, 17, 18]]])

print(a.shape)

#参数: 
# tf.split(value, num_or_size_splits, axis)
# value : tensor
# num_or_size_splits : 维度大小
# axis : 哪个轴划分

b_list = []
b = tf.split(a, 2, 1)
for b_elem in b:
    #Tensor("split:0", shape=(3, 1, 3), dtype=int32)
    print("old_b_elem:", b_elem)

    #Tensor("Squeeze:0", shape=(3, 3), dtype=int32)
    b_elem = tf.squeeze(b_elem)
    print("new_b_elem:", b_elem)

    b_list.append(b_elem)


#[[ 1  2  3]
# [ 7  8  9]
# [13 14 15]]
# [[ 4  5  6]
#  [10 11 12]
#  [16 17 18]]
with tf.Session() as sess:
    for elem in b_list:
        print(sess.run(elem))
