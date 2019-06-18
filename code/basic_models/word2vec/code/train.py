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

print(u_emb)

pos_u_emb = tf.matmul(pos_u, u_emb)
pos_v_emb = tf.matmul(pos_v, v_emb)
neg_v_emb = tf.matmul(neg_v, v_emb)


print(pos_u_emb)
print(pos_v_emb)
print(neg_v_emb)

#with tf.Session() as sess:

