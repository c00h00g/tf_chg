import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

"""
任务：使用lr模型对图片进行分类
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)

#定义输入输出

#需要feed数据的变量
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


#定义基础参数
learning_rate = 0.01

#batch大小
batch_size = 100

#10轮
epochs = 25

#定义具体操作
pred = tf.nn.softmax(tf.matmul(x, w) + b)

#定义优化目标
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        avg_loss = 0
        batch_num = int(mnist.train.num_examples / batch_size)
        for i in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, total_loss = sess.run([optimizer, loss], feed_dict = {x : batch_x, y : batch_y})
            avg_loss += total_loss / batch_num

        print("train loss is : ", avg_loss)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    precision = sess.run([accuracy], feed_dict = {x : mnist.test.images, y : mnist.test.labels})
    print('precision is: ', precision[0])



