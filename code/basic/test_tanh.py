import tensorflow as tf
import math

def calc_tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

#0.761594155956
print calc_tanh(1)
    

a = tf.constant([1, 1, 1], dtype = tf.float32)
b = tf.tanh(a)

with tf.Session() as sess:
    #[0.7615942 0.7615942 0.7615942]
    print(sess.run(b))
