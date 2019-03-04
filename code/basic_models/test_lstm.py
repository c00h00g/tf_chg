
# -*- coding:utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data', one_hot = True)


n_units = 128
n_classes = 10
batch_size = 100
n_steps = 28
learning_rate = 0.001
epochs = 20

x = tf.placeholder(tf.float32, [None, n_steps * n_steps])
y = tf.placeholder(tf.float32, [None, n_classes])

re_x = tf.reshape(x, [-1, n_steps, n_steps])


lstm_cell = tf.nn.rnn_cell.LSTMCell(n_units)
initial_state = lstm_cell.zero_state(batch_size, dtype = tf.float32)

outputs, state = tf.nn.dynamic_rnn(lstm_cell, re_x, initial_state=initial_state, dtype = tf.float32)

weights = tf.Variable(tf.random_normal([n_units, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

# (, n_units)
last_output = outputs[:, -1, :]

logits = tf.matmul(last_output, weights) + b
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for one_epoch in range(epochs):
        batch_num = int(mnist.train.num_examples / batch_size)
        
        total_loss = 0
        for i in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict = {x : batch_x, y : batch_y})
            total_loss += loss
            
        print("train_loss is : ", total_loss / batch_num)
        print("train acc is : ", acc)

    test_acc = 0
    batch_num = int(mnist.test.num_examples / batch_size)
    for i in range(batch_num):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        acc = sess.run([accuracy], feed_dict = {x : batch_x, y : batch_y})
        test_acc += acc[0]
    print("test acc is : ", test_acc / batch_num)

#train_loss is :  1.716630594513633
#train acc is :  0.91
#train_loss is :  1.5205500355633823
#train acc is :  0.95
#train_loss is :  1.506055129441348
#train acc is :  0.96
#train_loss is :  1.4966566449945624
#train acc is :  1.0
#train_loss is :  1.4939226805080068
#train acc is :  0.94
#train_loss is :  1.4884288111600008
#train acc is :  0.95
#train_loss is :  1.48549202745611
#train acc is :  0.98
#train_loss is :  1.4835152103684166
#train acc is :  0.98
#train_loss is :  1.4829890268499202
#train acc is :  0.97
#train_loss is :  1.4817238311334089
#train acc is :  0.97
#train_loss is :  1.480152035409754
#train acc is :  1.0
#train_loss is :  1.4780623745918273
#train acc is :  1.0
#train_loss is :  1.477855222008445
#train acc is :  0.96
#train_loss is :  1.476537529555234
#train acc is :  0.97
#train_loss is :  1.4751309141245754
#train acc is :  1.0
#train_loss is :  1.4742528564279729
#train acc is :  1.0
#train_loss is :  1.4745444356311452
#train acc is :  0.99
#train_loss is :  1.473518825227564
#train acc is :  1.0
#train_loss is :  1.4730200984261252
#train acc is :  1.0
#train_loss is :  1.4718210176988082
#train acc is :  0.99
#test acc is :  0.9850000101327896
