# -*- coding:utf-8 -*-

import tensorflow as tf

def test_get_variable(sess):
    # [[1. 1. 1.]
    #  [1. 1. 1.]]
    a = tf.get_variable(name = 'chg', shape = [2, 3], initializer = tf.ones_initializer())
    #<tf.Variable 'chg:0' shape=(2, 3) dtype=float32_ref>
    print a

    #ValueError: Variable chg already exists, disallowed, 直接调用会报错
    b = tf.get_variable(name = 'chg')
    print b

    print sess.run(a)

def use_variable_scope(sess):
    """
    scope相当于命名空间的意思
    """
    with tf.variable_scope("scope"):
        a = tf.get_variable(name = 'chg', shape = [2, 3], initializer = tf.ones_initializer())
        #scope/chg:0
        print a.name

    with tf.variable_scope("scope", reuse = True):
        b = tf.get_variable(name = 'chg')
        #scope/chg:0
        print b.name

    #equal
    print "equal" if a == b else "not_equal"

with tf.Session() as sess:
    # 有变量的时候，必须初始化
    sess.run(tf.initialize_all_variables())
    #test_get_variable(sess)

    use_variable_scope(sess)
