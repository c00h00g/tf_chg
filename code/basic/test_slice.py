

import tensorflow as tf
import numpy as np

# tf.slice(inputs, begin, size)
# begin : 每个坐标的开始位置
# size : 每个坐标的长度

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = a[:, 0:2]
c = a[:, 2:]


s = np.random.rand(3, 4)
s_t = tf.convert_to_tensor(s)

# d表示从s_t维度0从0开始取两个元素，维度1从1开始取两个元素
d = tf.slice(s_t, [0, 1], [2, 2])


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

    print(sess.run(s_t))
    print(sess.run(d))

#outputs
#[[1 2 3]
# [4 5 6]]

#[[1 2]
# [4 5]]

#[[3]
# [6]]

#[[0.82735887 0.64161663 0.36697804 0.98156343]
# [0.73893879 0.65447965 0.57633745 0.37217261]
# [0.61845618 0.54796085 0.49454364 0.88034836]]

#[[0.64161663 0.36697804]
# [0.65447965 0.57633745]]
