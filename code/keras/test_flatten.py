
"""
Flattens the input. Does not affect the batch size.
在非batch层展开
"""
import tensorflow as tf
import numpy as np

a = np.random.randint(0, 10, size=(3, 4, 5))
a_t = tf.convert_to_tensor(a)


b = tf.keras.layers.Flatten()(a_t)


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b))
