import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


a = tf.constant([[1, 2, 3, 4], [9, 10, 11, 12]], dtype=tf.float32)
b = tf.constant([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=tf.float32)
print(a.shape)
print(b.shape)

ll = []
ll.append(a)
ll.append(b)

c = tf.reduce_mean(ll, axis = 0)
d = tf.reduce_mean(ll, axis = 1)

#[[ 3.  4.  5.  6.]
# [ 9. 10. 11. 12.]]
# [[ 5.  6.  7.  8.]
#  [ 7.  8.  9. 10.]]
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))

