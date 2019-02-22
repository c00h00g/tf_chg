import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)

#(55000, 784)
#(55000, 10)
#(5000, 784)
#(5000, 10)
#(10000, 784)
#(10000, 10)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)


#55000, 样本个数
print(mnist.train.num_examples)


#读取batch数据
#(1, 784)
#(1, 10)
x, y = mnist.train.next_batch(1)
print(x.shape)
print(y.shape)
