import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


#[[ 2.1079545  -0.6007467   1.4656574 ]
# [-0.48197764 -2.0265532   0.5914698 ]
# [-0.02484042 -0.28389436  1.0420202 ]]
# 生成正太分布
a = tf.random_normal([3, 3])

b = tf.Variable(a)

#有变量的时候必须先执行初始化操作，否则会报错
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
