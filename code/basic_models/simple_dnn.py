import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot=True)

input_size = 784
hidden_size_1 = 512
hidden_size_2 = 256
n_classes = 10

x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([input_size, hidden_size_1])),
    'h2': tf.Variable(tf.random_normal([hidden_size_1, hidden_size_2])),
    'out': tf.Variable(tf.random_normal([hidden_size_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([hidden_size_1])),
    'b2': tf.Variable(tf.random_normal([hidden_size_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#def simple_dnn(input):
#    """定义一个dnn网络"""
#    layer_1 = tf.add(tf.matmul(input, weights['h1']),  biases['b1'])
#    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),  biases['b2'])
#    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#    return out_layer


#超参数
learning_rate = 0.01
batch_size = 200
epochs = 200

#prediction

layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['h1']),  biases['b1']))
layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weights['h2']),  biases['b2']))
logits = tf.nn.leaky_relu(tf.matmul(layer_2, weights['out']) + biases['out'])

#logits = simple_dnn(x)
prediction = tf.nn.softmax(logits)

#loss
#这样写有问题，比如prediction==0的时候就异常
#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), 1))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))

#优化目标
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#train_op = optimizer.minimize(loss)

#evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        batch_num = int(mnist.train.num_examples / batch_size)

        all_loss = 0.0
        for i in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss_value = sess.run([train_op, loss], feed_dict = {x : batch_x, y : batch_y})
            all_loss += loss_value
            #print(sess.run([prediction, y], feed_dict = {x: batch_x, y: batch_y}))
            #print(sess.run(loss, feed_dict = {x : batch_x, y: batch_y}))
            #break

        print("loss is : ", all_loss)
    print("acc is:", sess.run(accuracy, feed_dict = {x : mnist.test.images, y: mnist.test.labels}))

