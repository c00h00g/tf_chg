import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


a = tf.random_normal([4, 4], dtype = tf.float32)
b = tf.nn.embedding_lookup(a, [[1, 2, 3], [1, 2, 3]])

#[array([[ 0.5978104 , -1.4587156 , -0.98457426,  0.1604308 ],
#        [ 0.11015393, -1.5336986 , -0.66205037, -0.9524743 ],
#        [ 0.19179282, -1.4519409 , -1.2713407 ,  0.06341794],
#        [ 0.0381289 , -2.0307865 , -0.8476785 , -1.1042027 ]],
#        dtype=float32), array([[[ 0.11015393, -1.5336986 , -0.66205037, -0.9524743 ],
#            [ 0.19179282, -1.4519409 , -1.2713407 ,  0.06341794],
#            [ 0.0381289 , -2.0307865 , -0.8476785 , -1.1042027 ]],
#
#            [[ 0.11015393, -1.5336986 , -0.66205037, -0.9524743 ],
#            [ 0.19179282, -1.4519409 , -1.2713407 ,  0.06341794],
#            [ 0.0381289 , -2.0307865 , -0.8476785 , -1.1042027 ]]],
#                                                                    dtype=float32)]

with tf.Session() as sess:
    print(sess.run([a, b]))


