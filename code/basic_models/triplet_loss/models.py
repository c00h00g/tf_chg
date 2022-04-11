# -*- coding:utf-8 -*-
# https://omoindrot.github.io/triplet-loss

import tensorflow as tf
import numpy as np

def pairwise_distances(embeddings, squared, sess):
    """
    Args:
        embeddings: tensor of shape (batch, embed_dim)
        squared: Boolean
    """
    # (batch, batch)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    print("chg dot_product is ------>")
    print(sess.run(dot_product))

    # (batch,)
    square_norm = tf.diag_part(dot_product)
    print("chg square_norm is ------>")
    print(sess.run(square_norm))

    # (1, batch) - (batch, batch) + (batch, 1)
    # 广播
    # (batch, batch)
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    print("chg distances is ------>")
    print(sess.run(distances))

    distances = tf.maximum(distances, 0)
    print("chg equal distances --->")
    print(sess.run(tf.equal(distances, 0)))

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances



if __name__ == '__main__':
    a = np.random.randint(0, 3, size=(3, 3))
    a_t = tf.convert_to_tensor(a, dtype=tf.float32)

    with tf.Session() as sess:
        print(sess.run(a_t))
        dis = pairwise_distances(a_t, True, sess)

######
"""
[[2. 1. 0.]
 [2. 2. 0.]
 [1. 0. 0.]]
chg dot_product is ------>
[[5. 6. 2.]
 [6. 8. 2.]
 [2. 2. 1.]]

chg square_norm is ------>
[5. 8. 1.]
chg distances is ------>
[[0. 1. 2.]
 [1. 0. 5.]
 [2. 5. 0.]]
chg equal distances --->
[[ True False False]
 [False  True False]
 [False False  True]]
"""
