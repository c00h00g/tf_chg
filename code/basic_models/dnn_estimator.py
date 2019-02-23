# -*- coding:utf-8 -*-

import sys
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data')


from tensorflow.python.platform import tf_logging
tf_logging.set_verbosity('INFO')

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("do_train", False, "whether to run training")
flags.DEFINE_bool("do_test", False, "whether to run test")
flags.DEFINE_bool("do_infer", False, "whether to run test")

def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

print(input(mnist.train)[0])
print(input(mnist.train)[0].shape)

print(input(mnist.train)[1])
print(input(mnist.train)[1].shape)

#sys.exit(0)

# 单特征多个维度需要指定
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]
# or feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[256, 32],
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=10,
        dropout=0.1,
        model_dir="./tmp/mnist_model"
)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": input(mnist.train)[0]},
        y=input(mnist.train)[1],
        num_epochs=None,
        batch_size=50,
        shuffle=True
)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": input(mnist.test)[0]},
        y=input(mnist.test)[1],
        num_epochs=1,
        shuffle=False
)

def serving_input_fn():
    images = tf.placeholder(tf.int32, [None, 28, 28], name = 'x')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({"x" : images})()
    return input_fn

#执行do_train
if FLAGS.do_train:
    classifier.train(input_fn=train_input_fn, steps=100000)

# Evaluate accuracy
if FLAGS.do_test:
    # ('evaluate_metrics:', {'average_loss': 0.07996486, 'accuracy': 0.983, 'global_step': 100000, 'loss': 10.122134})
    metrics = classifier.evaluate(input_fn=test_input_fn)
    print("evaluate_metrics:", metrics)

# output the model
if FLAGS.do_infer:
    classifier._export_to_tpu = False
    classifier.export_savedmodel("./export/", serving_input_fn)

