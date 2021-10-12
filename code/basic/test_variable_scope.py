
# tf.variable_scope()会在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
import tensorflow as tf

with tf.variable_scope('V1',reuse=None):
    a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a2 = tf.Variable(tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')

with tf.variable_scope('V2',reuse=tf.AUTO_REUSE):
    a3 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a4 = tf.Variable(tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print (a1.name)
    print (a2.name)
    print (a3.name)
    print (a4.name)

V1/a1:0
V1/a2:0
V2/a1:0
V2/a2:0
