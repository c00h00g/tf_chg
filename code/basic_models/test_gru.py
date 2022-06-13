# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

input_data = np.random.randint(0, 10, size=(3, 5, 20))
input_t = tf.convert_to_tensor(input_data, dtype=tf.float32)


gru_cell = tf.nn.rnn_cell.GRUCell(num_units=128)


# outputs: 表示每个cell的输出
# last_state: 表示最终的状态
outputs, last_state = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input_t, dtype=tf.float32)

print(gru_cell)
print(outputs)
print(last_state)

"""
Tensor("rnn/transpose_1:0", shape=(3, 5, 128), dtype=float32)
Tensor("rnn/while/Exit_3:0", shape=(3, 128), dtype=float32)
"""
