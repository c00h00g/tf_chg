import numpy as np
import tensorflow as tf
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_dict = dict()

def make_one_hot(input, max_depth = 3):
    res = []
    for i in range(max_depth):
        if i + 1 == input:
            res.append(1)
        else:
            res.append(0)
    return res

def load_f_data(path):
    f_list = []
    label_list = []

    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip()
            line_sp = line.split('\t')
            f_list.append([float(x) for x in line_sp[0:4]])
            label = int(line_sp[4])
            label_list.append(make_one_hot(label))
    return np.array(f_list), np.array(label_list)

x_train, x_label = load_f_data('train.tsf')
x_test, y_test = load_f_data('test.tsf')
print x_train.shape
print x_label.shape
#3sys.exit(0)

x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None, 3])

w = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))

learning_rate = 0.01

epochs = 30000

init = tf.global_variables_initializer()

pred = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        _, total_loss = sess.run([optimizer, loss], feed_dict = {x : x_train, y: x_label})
        print('total loss is : %s', total_loss)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    precision = sess.run([accuracy], feed_dict = {x : x_test, y : y_test})

    print('precision is: ', precision[0])





