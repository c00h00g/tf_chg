# -*- coding:utf-8 -*-
# 功能和python的功能一致

import tensorflow as tf

seq_len = 10

# [0 1 2 3 4 5 6 7 8 9]
# 参数 
# start
# limit
# delta
a = tf.range(0, 10, delta = 1)

# [0 2 4 6 8]
b = tf.range(0, 10, delta = 2)

c = a * seq_len
print c

#[[ 0]
# [10]
# [20]
# [30]
# [40]
# [50]
# [60]
# [70]
# [80]
# [90]]
# [-1,. 1] 表示reshape成两维的
d = tf.reshape(a * seq_len, [-1, 1])

with tf.Session() as sess:
    print sess.run(a)
    print sess.run(b)

    print sess.run(c)
    print sess.run(d)
