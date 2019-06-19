#t -*- coding:utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_data():
    """
    load train data
    """
    all_data = []
    with open('input_data') as f:
        for line in f.readlines():
            line = line.rstrip()
            line_sp = line.split(' ')
            all_data.append(line_sp)
    return all_data

def get_batch_data(data_list, start_idx, batch_size):
    """
    get a batch data
    """
    data_len = len(data_list)
    next_start = -1
    key_before = ""
    # center word count
    num = 0
    res_list = []

    i = -1
    for i in range(start_idx, data_len):
        key = data_list[i][0]
        if key != key_before:
            if num == batch_size:
                next_start = i
                break
            res_list.append(data_list[i])
            key_before = key
            num += 1
        else:
            res_list.append(data_list[i])

    if i == -1 or i == data_len - 1:
        next_start = -1

    return next_start, np.array(res_list)

def dump_embdding(emb):
    row, col = emb.shape
    f = open('emb_res', 'w')
    for i in range(row):
        one_row = []
        for j in range(col):
            one_row.append(str(emb[i][j]))
        write_line = ' '.join(one_row) + '\n'
        f.write(write_line)
    f.close()

################### load data #####################
all_data = load_data()

epochs = 20
batch_size = 100
emb_size = 100
vocab_size = 19188

x = tf.placeholder(tf.int32, [None, 7])

pos_u = tf.slice(x, [0, 0], [-1, 1])
pos_v = tf.slice(x, [0, 1], [-1, 1])
neg_v = tf.slice(x, [0, 2], [-1, 5])

pos_u = tf.one_hot(pos_u, depth = vocab_size, dtype = tf.float32)
pos_v = tf.one_hot(pos_v, depth = vocab_size, dtype = tf.float32)
neg_v = tf.one_hot(neg_v, depth = vocab_size, dtype = tf.float32)

pos_u = tf.reshape(pos_u, [-1, vocab_size])
pos_v = tf.reshape(pos_v, [-1, vocab_size])
neg_v = tf.reshape(neg_v, [-1, vocab_size])

print(pos_u)
print(pos_v)
print(neg_v)

#two kinds of embedding
u_emb = tf.Variable(tf.random_normal([vocab_size, emb_size]))
v_emb = tf.Variable(tf.random_normal([vocab_size, emb_size]))

# trans to emb
pos_u_emb = tf.reshape(tf.matmul(pos_u, u_emb), [-1, 1, 100])
pos_v_emb = tf.reshape(tf.matmul(pos_v, v_emb),  [-1, 1, 100])
neg_v_emb = tf.reshape(tf.matmul(neg_v, v_emb), [-1, 5, 100])

print(pos_u_emb)
print(pos_v_emb)
print(neg_v_emb)

pos_loss = tf.reduce_sum(tf.log_sigmoid(tf.reduce_sum(tf.squeeze(pos_u_emb * pos_v_emb, [1]), [1])))
neg_loss = tf.reduce_sum(tf.log_sigmoid(-tf.reduce_sum(pos_u_emb * neg_v_emb, [2])))

loss = -(pos_loss + neg_loss)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train_op = optimizer.minimize(loss)

print(pos_loss)
print(neg_loss)
print(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        start_idx = 0
        num = 0
        while True:
            new_idx, batch_data = get_batch_data(all_data, start_idx, batch_size)
            if new_idx == -1:
                break
            start_idx = new_idx

            _, loss_value = sess.run([train_op, loss], feed_dict = {x : batch_data})

            num += 1

            if num % 100 == 0:
                print('loss is : ', loss_value)
    # output
    emb_res = u_emb.eval()
    dump_embdding(emb_res)
    print('dump data finish!')





