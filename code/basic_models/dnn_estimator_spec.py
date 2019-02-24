import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from tensorflow.python.platform import tf_logging
tf_logging.set_verbosity('INFO')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot=False)

#parameters
learning_rate = 1e-4
batch_size = 200
n_hidden_1 = 256
n_hidden_2 = 256
epochs = 10000

n_input = 784
n_classes = 10

input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"images": mnist.train.images}, y = mnist.train.labels,
    batch_size = batch_size,
    num_epochs = None,
    shuffle = True
)

def neural_net(x_dict):
    x = x_dict['images']
    layer_1 = tf.layers.dense(x, n_hidden_1, activation = tf.nn.leaky_relu)
    layer_1 = tf.layers.dropout(layer_1, rate = 0.2)

    layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation = tf.nn.leaky_relu)
    layer_2 = tf.layers.dropout(layer_2, rate = 0.2)

    out_layer = tf.layers.dense(layer_2, n_classes, activation = tf.nn.leaky_relu)
    return out_layer


def model_fn(features, labels, mode):
    logits = neural_net(features)
    pred_classes = tf.argmax(logits, axis = 1)
    pred_prob = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions = pred_classes)

    #为什么要用sparse
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

    #效果比较差
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

    #accuracy = 0.9743, 优化效果比较好
    #optimizer = tf.train.AdamOptimizer(1e-4)

    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(mode = mode, predictions = pred_classes, loss = loss_op, train_op = train_op, eval_metric_ops = {'accuracy' : acc_op})
    return estim_specs

#create the model
model = tf.estimator.Estimator(model_fn)
model.train(input_fn, steps=epochs)


#evaluate the model
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
model.evaluate(input_fn)

