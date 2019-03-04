# -*- coding:utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data', one_hot = True)

batch_size = 25
n_units = 128
n_steps = 28
n_epochs = 100
n_classes = 10
learning_rate = 0.001

fw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_units, forget_bias=1.0, state_is_tuple=True)
fw_cell.zero_state(batch_size, dtype = tf.float32)

bw_cell = tf.nn.rnn_cell.LSTMCell(num_units = n_units, forget_bias=1.0, state_is_tuple=True)
bw_cell.zero_state(batch_size, dtype = tf.float32)

weights = tf.Variable(tf.random_normal([2 * n_units, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

x = tf.placeholder(tf.float32, [None, n_steps * n_steps])
bi_x = tf.reshape(x, [-1, n_steps, n_steps])

y = tf.placeholder(tf.float32, [None, n_classes])

outputs, final_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, bi_x, dtype = tf.float32)

# 将正向和反向连接在一起
outputs_concat = tf.concat(outputs, 2)

# (None, 256)
last_cell = outputs_concat[:, -1, :]

# (?, 10)
logits = tf.matmul(last_cell, weights) + b

#(?, 10)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        batch_num = int(mnist.train.num_examples / batch_size)

        total_loss = 0
        for i in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss, acc = sess.run([train_op, loss_op, accuarcy], feed_dict = {x : batch_x, y : batch_y})

            total_loss += loss
            if i % 200 == 1:
                print("train loss is : %s", total_loss / (i + 1))
                print("train acc is : %s", acc)

    test_acc = sess.run([accuarcy], feed_dict = {x : mnist.test.images, y : mnist.test.labels})
    print("test acc is : ", test_acc)
