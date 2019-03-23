import tensorflow as tf
tf.enable_eager_execution()

#tf.Tensor(
#[[1 1 1 1]
# [1 1 1 1]], shape=(2, 4), dtype=int32)

a = tf.ones(shape = [2, 4], dtype = tf.int32)
print(a)
