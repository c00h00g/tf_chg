import tensorflow as tf
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])  


with tf.Session() as sess:
    #按列求和
    #[5 7 9]
    print(sess.run(tf.reduce_sum(a, 0)))

    #按行求和
    #[6, 15]
    print(sess.run(tf.reduce_sum(a, 1)))

