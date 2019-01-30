import tensorflow as tf

classes = 3
labels = tf.constant([1, 2, 3])
labels_2 = tf.constant([1, 2, 0])

output = tf.one_hot(labels, classes)
output_2 = tf.one_hot(labels_2, classes)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
    print(sess.run(output_2))


#输出结果，可以看出编码最大值是classes - 1, 超过的部分无法编码出来
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]]

[[0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]]
