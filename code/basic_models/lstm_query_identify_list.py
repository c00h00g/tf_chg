# -*- coding:utf-8 -*-
"""
使用lstm模型进行query分类
"""
import sys
import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_seq_length", 20, "Query indentify max length")

flags.DEFINE_integer("train_batch_size", 20, "Total batch size for training.")

flags.DEFINE_integer("vocab_size", 410004, "Total batch size for training.")

flags.DEFINE_integer("emb_size", 100, "Total batch size for training.")

flags.DEFINE_string("data_dir", "../data/", "The input data dir!")

flags.DEFINE_string("output_dir", "./output_dir/", "The input data dir!")

flags.DEFINE_string("model_dir", "./model_dir/", "The input data dir!")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run training.")

flags.DEFINE_integer("hidden_units", 100, "Whether to run training.")

flags.DEFINE_integer("num_epochs", 10, "How many epochs to train.")


class DataProcessor(object):
    def __init__(self):
        return

    def get_train_examples(self, data_dir):
        return self._read_data(data_dir + "/train.txt")

    def get_dev_examples(self, data_dir):
        return self._read_data(data_dir + "/dev.txt")

    def get_test_examples(self, data_dir):
        return self._read_data(data_dir + "/test.txt")

    def _read_data(self, data_dir):
        """
        读取数据转换成id
        """
        feature_list = []
        label_list = []
        with open(data_dir) as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines))):
                line = lines[i].rstrip()
                line_sp = line.split('\t')
                q_ids = line_sp[0]
                label = int(line_sp[1])

                q_ids_sp = q_ids.split(' ')
                q_ids_sp = [int(i) + 1 for i in q_ids_sp]

                #max_len = FLAGS.max_seq_length if len(q_ids_sp) > FLAGS.max_seq_length else FLAGS.max_seq_length

                #使用0做padding
                for i in range(len(q_ids_sp), FLAGS.max_seq_length):
                    q_ids_sp.append(0)

                feature_list.append(q_ids_sp[0:20])
                label_list.append(label)

        #补齐剩余长度
        #print(feature_list)
        return (feature_list, label_list)

def file_based_convert_examples_to_features(examples, max_seq_length, output_file):
    """
    将输入数据转化为tfrecord格式
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    length = len(examples[0])

    for i in tqdm(range(length)):
        train_example = tf.train.Example(features = tf.train.Features(feature = {
            "ids" : tf.train.Feature(
                int64_list = tf.train.Int64List(value = examples[0][i])),

            "labels": tf.train.Feature(
                int64_list = tf.train.Int64List(value = [examples[1][i]])),
        }))
        #print(train_example)
        writer.write(train_example.SerializeToString())
        if i % 50000 == 0:
            tf.logging.info("deal %s cases" %(i))
    writer.close()

def read_data(self, input):
    """
    读取特征和label
    """
    feature_list = []
    label_list = []
    with open(input) as f:
        for line in f.readlines():
            line = line.rstrip()
            line_sp = line.split('\t')
            q_ids = line_sp[0]
            label = int(line_sp[1])

            q_ids_sp = q_ids.split(' ')
            q_ids_sp = [int(i) + 1 for i in q_ids_sp]

            #使用0做padding
            for i in range(len(q_ids_sp), FLAGS.max_seq_length):
                q_ids_sp.append(0)

            feature_list.append(q_ids_sp)
            label_list.append(label)

    #补齐剩余长度
    #print(feature_list)
    return (feature_list, label_list)

def build_train_examples(feature_label):
    """
    构造训练的tf_record
    """
    writer = tf.python_io.TFRecordWriter("train.TFRecord")

    length = len(feature_label[0])
    for i in range(length):
        train_example = tf.train.Example(features = tf.train.Features(feature = {
            "ids" : tf.train.Feature(
                int64_list = tf.train.Int64List(value = feature_label[0][i])),

            "labels": tf.train.Feature(
                int64_list = tf.train.Int64List(value = [feature_label[1][i]])),
        }))
        #print(train_example)
        writer.write(train_example.SerializeToString())
    writer.close()

def parse_example(serialized_example):
    """
    读取tfRecord数据
    """
    #filename_queue = tf.train.string_input_producer(["query_identify.TFRecord"])
    #reader = tf.TFRecordReader()
    #_, serialized_example = reader.read(filename_queue)

    name_to_features = {
        "ids" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "labels" : tf.FixedLenFeature([1], tf.int64),
    }

    example = tf.parse_single_example(serialized_example, name_to_features);

    for name in list(name_to_features.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    print("examples : ", example)
    return example

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, batch_size = 20):
    """
    将tfrecord转换为estimator的输入
    """
    name_to_features = {
        "ids" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "labels" : tf.FixedLenFeature([1], tf.int64),
    }

    def parser(record):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(name_to_features.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    #dataset = tf.data.TFRecordDataset(input_file)
    #dataset = dataset.map(parser)
    #dataset = dataset.shuffle(1000).repeat().batch(batch_size, drop_remainder = True)
    #print("dataset is : ", dataset)
    #print("batch_size is : ", batch_size)
    #dataset = dataset.make_one_shot_iterator().get_next()
    #return dataset['ids'], dataset['labels']

    def input_fn(params):
        dataset = tf.data.TFRecordDataset(input_file)

        print("chg-params:", params)

        dataset = dataset.map(parser)
        dataset = dataset.shuffle(1000).repeat().batch(params['batch_size'], drop_remainder = True)
        print("dataset is : ", dataset)

        dataset = dataset.make_one_shot_iterator().get_next()
        return dataset['ids'], dataset['labels']
    return input_fn

        #if is_training:
        #    dataset = dataset.shuffle(buffer_size = 100)

        #dataset = dataset.batch(18)
        #dataset = dataset.prefetch(1)

        #iterator = dataset.make_one_shot_iterator()
        #features, target = iterator.get_next()

        #print("dataset:", dataset)

        #dataset = dataset.apply(
        #    tf.contrib.data.map_and_batch(
        #        lambda record : parser(record, name_to_features),
        #        #batch_size = params['batch_size'],
        #        batch_size = 20,
        #        drop_remainder = drop_remainder))
        #return features, target


def lstm_model_fn(features, labels, mode, params):
    """
    构造lstm model
    """
    print("chg_params:", params)
    lstm_cell = rnn.LSTMCell(
        num_units = params['hidden_units'],
        forget_bias = params['forget_bias'],
        activation = tf.nn.tanh,
    )

    print("chg_features:", features)
    print("chg_labels:", labels)

    inputs = tf.cast(features, dtype = tf.int32)
    #inputs = features[0]

    label_ids = tf.cast(labels, dtype = tf.int32)
    #label_ids = features[1]

    print("mode inputs:", inputs)
    print("mode label_ids:", label_ids)
    print("mode 1:", mode)

    embedding = tf.Variable(tf.random_normal([FLAGS.vocab_size, FLAGS.emb_size]))
    #(200, 20, 100)
    inputs = tf.nn.embedding_lookup(embedding, inputs)
    print("inputs shape 1:", inputs)

    inputs = tf.split(inputs, FLAGS.max_seq_length, 1)
    #print("inputs shape 2:", inputs.shape)
    print("inputs shape 2:", inputs)
    print("inputs shape 2:", type(inputs))


    for i in range(len(inputs)):
        inputs[i] = tf.squeeze(inputs[i])

    print("intputs shape 3:", inputs)

    #inputs = tf.squeeze(inputs)
    #print("inputs shape 3:", (inputs))

    outputs, _ = rnn.static_rnn(cell = lstm_cell, inputs = inputs, dtype = tf.float32)
    outputs = outputs[-1]
    print(outputs)
    hidden_size = outputs.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02),
    )

    output_bias = tf.get_variable(
        "output_bias", [2],
        initializer=tf.zeros_initializer(),
    )

    logits = tf.matmul(outputs, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, -1)
    log_probs = tf.nn.log_softmax(logits, -1)

    one_hot_labels = tf.one_hot(label_ids, depth = 2, dtype = tf.float32)
    one_hot_labels = tf.squeeze(one_hot_labels)
    print("one_hot:", one_hot_labels)
    print("log probs", log_probs)

    per_example_loss = -tf.reduce_mean(one_hot_labels * log_probs, axis = -1)
    total_loss = tf.reduce_mean(per_example_loss)
    print("per loss", per_example_loss)
    print("total loss", total_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)

        train_op = optimizer.minimize(loss = total_loss, global_step=tf.train.get_global_step())

        logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)


        output_spec = tf.estimator.EstimatorSpec(
                          mode=mode,
                          loss=total_loss,
                          train_op=train_op,
                          training_hooks=[logging_hook])

        return output_spec

    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(logits, axis = -1, output_type = tf.int32)
        accuracy = tf.metrics.accuracy(
            labels = label_ids,
            predictions = predictions,
        )

        loss = tf.metrics.mean(values = per_example_loss)

        eval_metrics = {
            'eval_accuracy' : accuracy,
        }

        output_spec = tf.estimator.EstimatorSpec(
            mode = mode,
            loss = total_loss, 
            eval_metric_ops = eval_metrics,
        )

        return output_spec

def debug_dataset(dataset):
    """
    打印dataset
    """
    iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(iterator.get_next()))
            except:
                break

def test_tfrecord_data():
    d = tf.data.TFRecordDataset("output_dir/train.TFRecord")
    dataset = d.map(parse_example)
    dataset = dataset.batch(10)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    data_processor = DataProcessor()
    train_examples = None

    print(FLAGS.output_dir)
    print(FLAGS.max_seq_length)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    if FLAGS.do_train:
        run_config = tf.estimator.RunConfig(
            tf_random_seed=19830610,
            model_dir=FLAGS.model_dir,
        )

    estimator = tf.estimator.Estimator(model_fn = lstm_model_fn, 
                                       model_dir = FLAGS.model_dir,
                                       params = {
                                           'batch_size' : FLAGS.train_batch_size,
                                           'num_epochs' : FLAGS.num_epochs,
                                           'hidden_units' : FLAGS.hidden_units,
                                           'learning_rate' : 0.0001,
                                           'forget_bias' : 1.0,
                                       })

    if FLAGS.do_train:
        #获取训练数据
        #train_examples = data_processor.get_train_examples(FLAGS.data_dir)
        train_file = os.path.join(FLAGS.output_dir, "train.TFRecord")
        #file_based_convert_examples_to_features(train_examples, FLAGS.max_seq_length, train_file)

        train_input_fn = file_based_input_fn_builder(
            input_file = train_file,
            seq_length = FLAGS.max_seq_length, 
            is_training = True,
            drop_remainder = True)

        print("train_input_fn:", train_input_fn)

        estimator.train(train_input_fn, max_steps = 10000)

        #estimator.train(input_fn = lambda : file_based_input_fn_builder(train_file, FLAGS.max_seq_length, is_training = True, drop_remainder = True), max_steps = 100)

    if FLAGS.do_eval:
        #dev_examples = data_processor.get_dev_examples(FLAGS.data_dir)
        dev_file = os.path.join(FLAGS.output_dir, "eval.TFRecord")
        #file_based_convert_examples_to_features(dev_examples, FLAGS.max_seq_length, dev_file)

        eval_input_fn = file_based_input_fn_builder(
            input_file = dev_file,
            seq_length = FLAGS.max_seq_length, 
            is_training = True,
            drop_remainder = False)

        result = estimator.evaluate(eval_input_fn, steps = 100)


if __name__ == "__main__":
    tf.app.run()
    #a = queryIdentifyLSTM("../data/data.tsv.utf8.2k")
    #feature_label = a.read_data()
    #a.build_train_examples(feature_label)
    #test_tfrecord_data()

