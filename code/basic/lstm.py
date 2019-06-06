# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = 128)
inputs = tf.placeholder(np.float32, shape = (32, 100))
h0 = lstm_cell.zero_state(32, np.float32)

#如何单步调用
output, h1 = lstm_cell.__call__(inputs, h0)

# lstm's output contains two states cell_state & hidden state

#Tensor("lstm_cell/mul_2:0", shape=(32, 128), dtype=float32)
#Tensor("lstm_cell/add_1:0", shape=(32, 128), dtype=float32)
#Tensor("lstm_cell/mul_2:0", shape=(32, 128), dtype=float32)
print(h1.h)
print(h1.c)
print(output)



