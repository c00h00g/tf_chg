import tensorflow as tf
import numpy as np

a = np.random.randint(0, 10, size = (2, 3, 4))
a_t = tf.convert_to_tensor(a)

shape = a_t.get_shape()
print("shape is : %s, type is : %s" % (shape, type(shape)))


with tf.Session() as sess:
    print(sess.run(a_t))


"""
shape is : (2, 3, 4), type is : <class 'tensorflow.python.framework.tensor_shape.TensorShape'>

2022-01-30 22:48:30.750063: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
[[[8 2 5 3]
  [9 2 1 2]
  [6 2 6 2]]

 [[3 7 4 6]
  [0 8 1 8]
  [4 1 4 5]]]
"""
