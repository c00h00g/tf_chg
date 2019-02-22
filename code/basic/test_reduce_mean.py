import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)


with tf.Session() as sess:
    # 2.5
    print(sess.run(tf.reduce_mean(a)))

    # [2. 3.] 求每列的平均
    print(sess.run(tf.reduce_mean(a, 0)))

    # [1.5 3.5], 求每行的平均
    print(sess.run(tf.reduce_mean(a, 1)))
