import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data', one_hot = True)


#train_num :  55000
print("train_num : ", mnist.train.num_examples)
#validation_num :  5000
print("validation_num : ", mnist.validation.num_examples)
#test_num :  10000
print("test_num : ", mnist.test.num_examples)

#(55000, 784)
print(mnist.train.images.shape)

# (55000,)
# (55000, 10) : 参数one_hot = True的时候
print(mnist.train.labels.shape)

# [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print(mnist.train.labels[0])


#如何获取batch
xs, ys = mnist.train.next_batch(10)

#(10, 784)
print(xs.shape)

#(10, 10)
print(ys.shape)
