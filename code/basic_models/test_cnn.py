# -*- coding:utf-8 -*-

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../data/", one_hot=True)

#batch_x, batch_y = mnist.train.next_batch(1)

x_input = tf.placeholder(tf.float32, [None, 28 * 28])
# batch_size * height * width * channel
x = tf.reshape(x_input, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

batch_size = 128 * 2
learning_rate = 0.0001
epochs = 100

# if padding == 'same', use 0 to padding
# if padding == 'valid', not padding
conv1 = tf.layers.conv2d(
            inputs = x,
            filters = 16,
            kernel_size = 5,
            strides = 1,
            padding = 'valid',
            activation = tf.nn.relu)

# Tensor("conv2d/Relu:0", shape=(?, 24, 24, 16), dtype=float32)
print(conv1)

# Tensor("max_pooling2d/MaxPool:0", shape=(?, 12, 12, 16), dtype=float32)
# 每次移动两步
pool1 = tf.layers.max_pooling2d(conv1, pool_size = 2, strides = 2)
print(pool1)

conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 32,
            kernel_size = 5,
            strides = 1,
            padding = 'valid',
            activation = tf.nn.relu)
# (?, 8, 8, 32)
print(conv2)

pool2 = tf.layers.max_pooling2d(conv2, pool_size = 2, strides = 2)
# (?, 4, 4, 32)
print(pool2)

flat = tf.layers.flatten(pool2)
# Tensor("flatten/Reshape:0", shape=(?, 512), dtype=float32)
print(flat)

output = tf.layers.dense(flat, 10)
prediction = tf.nn.softmax(output)

loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        batch_num = int(mnist.train.num_examples / batch_size)

        num = 0
        for j in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, loss = sess.run([optimizer, loss_op], feed_dict = {x_input : batch_x, y : batch_y})
            num += 1
            if num % 100 == 0:
                print('train loss is :', loss)
                acc = sess.run(accuracy, feed_dict = {x_input : mnist.test.images, y : mnist.test.labels})
                print('test acc is : ', acc)

