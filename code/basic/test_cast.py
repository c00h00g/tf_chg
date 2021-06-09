import tensorflow as tf
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


a = tf.constant([1, 2, 3, 4])

with tf.Session() as sess:
    #[1. 2. 3. 4.]
    print(sess.run(tf.cast(a, dtype = tf.float32)))

