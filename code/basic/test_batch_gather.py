# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

a = np.random.rand(3, 4, 5)
a_t = tf.convert_to_tensor(a)

b = np.random.randint(0, 4, size = (3, 2))
b_t = tf.convert_to_tensor(b)

c = tf.batch_gather(a_t, b_t)


with tf.Session() as sess:
    print(sess.run(a_t))
    print(sess.run(b_t))
    print(sess.run(c))


#[[[0.74126635 0.3223967  0.67246749 0.94245176 0.03275141]
#  [0.28085057 0.78391114 0.98578592 0.17047959 0.07586469]
#  [0.20340673 0.76148907 0.00956021 0.36726728 0.4987349 ]
#  [0.86450004 0.22389283 0.21793711 0.18573815 0.59965745]]
#
# [[0.46481599 0.16315235 0.16661557 0.34802409 0.52859948]
#  [0.80677437 0.51576828 0.34239105 0.09032319 0.42672308]
#  [0.13858404 0.74688882 0.70843271 0.57794411 0.47067383]
#  [0.01570147 0.04004877 0.5996255  0.16246137 0.95933844]]
#
# [[0.8767085  0.81752488 0.7885684  0.2779582  0.64031313]
#  [0.17202566 0.28657134 0.34399073 0.80496497 0.60436556]
#  [0.78153453 0.90403843 0.40290295 0.94404016 0.12934218]
#  [0.04625755 0.5403787  0.44719225 0.57900369 0.15079685]]]
#
#[[2 3]
# [0 0]
# [2 2]]
#
#[[[0.20340673 0.76148907 0.00956021 0.36726728 0.4987349 ]
#  [0.86450004 0.22389283 0.21793711 0.18573815 0.59965745]]
#
# [[0.46481599 0.16315235 0.16661557 0.34802409 0.52859948]
#  [0.46481599 0.16315235 0.16661557 0.34802409 0.52859948]]
#
# [[0.78153453 0.90403843 0.40290295 0.94404016 0.12934218]
#  [0.78153453 0.90403843 0.40290295 0.94404016 0.12934218]]]
