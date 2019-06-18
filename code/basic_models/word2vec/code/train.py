# -*- coding:utf-8 -*-

import tensorflow as tf

epochs = 1
batch_size = 128
emb_size = 100
vocab_size = 1000

x = tf.placeholder(tf.int32, [None, 6])

pos_u = tf.slice(x, [0, 0], [-1, 1])
pos_v = tf.slice(x, [0, 1], [-1, 1])
neg_v = tf.slice(x, [0, 2], [-1, 4])

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
neg_v_emb = tf.reshape(tf.matmul(neg_v, v_emb), [-1, 4, 100])

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




#with tf.Session() as sess:
