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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_seq_length", 25, "Query indentify max length")

flags.DEFINE_integer("train_batch_size", 20, "Total batch size for training.")

flags.DEFINE_integer("vocab_size", 37118, "Total batch size for training.")

flags.DEFINE_integer("emb_size", 256, "Total batch size for training.")

flags.DEFINE_string("data_dir", "../data/", "The input data dir!")

flags.DEFINE_string("output_dir", "./output_dir/", "The input data dir!")

flags.DEFINE_string("model_dir", "./model_dir/", "The input data dir!")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run training.")

flags.DEFINE_integer("hidden_units", 128, "Whether to run training.")

flags.DEFINE_integer("num_epochs", 10, "How many epochs to train.")


class DataProcessor(object):
    def __init__(self):
        return

    def get_train_examples(self, data_dir):
        return self._read_data(data_dir + "/hrs_train")

    def get_dev_examples(self, data_dir):
        return self._read_data(data_dir + "/hrs_eval")

    def get_test_examples(self, data_dir):
        return self._read_data(data_dir + "/train.txt")

    def _read_data(self, data_dir):
        """
        读取数据转换成id
        """
        feature_list1 = []
        feature_list2 = []
        label_list = []
        with open(data_dir) as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines))):
                line = lines[i].rstrip()
                line_sp = line.split('\t')
                q_ids1 = line_sp[0]
                q_ids2 = line_sp[1]
                label = int(line_sp[2])

                q_ids_sp1 = q_ids1.split(' ')
                q_ids_sp1 = [int(i) + 1 for i in q_ids_sp1]

                q_ids_sp2 = q_ids2.split(' ')
                q_ids_sp2 = [int(i) + 1 for i in q_ids_sp2]

                #使用0做padding
                for i in range(len(q_ids_sp1), FLAGS.max_seq_length):
                    q_ids_sp1.append(0)

                for i in range(len(q_ids_sp2), FLAGS.max_seq_length):
                    q_ids_sp2.append(0)

                feature_list1.append(q_ids_sp1[0:25])
                feature_list2.append(q_ids_sp2[0:25])
                label_list.append(label)

        #补齐剩余长度
        #print(feature_list1)
        #print(feature_list2)
        #print(label_list)
        return (feature_list1, feature_list2, label_list)

def file_based_convert_examples_to_features(examples, max_seq_length, output_file):
    """
    将输入数据转化为tfrecord格式
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    length = len(examples[0])

    for i in tqdm(range(length)):
        train_example = tf.train.Example(features = tf.train.Features(feature = {
            "ids1" : tf.train.Feature(
                int64_list = tf.train.Int64List(value = examples[0][i])),

            "ids2" : tf.train.Feature(
                int64_list = tf.train.Int64List(value = examples[1][i])),

            "labels": tf.train.Feature(
                int64_list = tf.train.Int64List(value = [examples[2][i]])),
        }))

        writer.write(train_example.SerializeToString())

        if i % 50000 == 0:
            tf.logging.info("deal %s cases" %(i))

    writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """
    将tfrecord转换为estimator的输入
    """
    name_to_features = {
        "ids1" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "ids2" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "labels" : tf.FixedLenFeature([], tf.int64),
    }

    def parser(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(name_to_features.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        dataset = tf.data.TFRecordDataset(input_file)
        dataset = dataset.shuffle(buffer_size = 10000)
        dataset = dataset.repeat()
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda record : parser(record, name_to_features),
                batch_size = params['batch_size'],
                drop_remainder = drop_remainder))
        return dataset

    return input_fn


def simnet_model_fn(features, labels, mode, params):
    """
    构造simnet model
    """
    print("chg_params:", params)

    inputs1 = tf.cast(features['ids1'], dtype = tf.int32)

    inputs2 = tf.cast(features['ids2'], dtype = tf.int32)

    label_ids = tf.cast(features['labels'], dtype = tf.int32)

    embedding = tf.Variable(tf.random_normal([FLAGS.vocab_size, FLAGS.emb_size]))

    inputs1 = tf.nn.embedding_lookup(embedding, inputs1)

    inputs2 = tf.nn.embedding_lookup(embedding, inputs2)

    # (20, 25, 100)
    print("inputs1 is :", inputs1)

    # (20, 25, 100)
    print("inputs2 is :", inputs2)

    outputs1 =  tf.reduce_sum(inputs1, 1)
    print("outputs1 is :", outputs1)

    # (20, 100)
    outputs2 =  tf.reduce_sum(inputs2, 1)
    print("outputs2 is :", outputs1)

    hidden_size = outputs1.shape[-1].value

    #(20, 200)
    outputs = tf.concat([outputs1, outputs2], 1)
    print("outputs 1 is : ", outputs)

    #(20, 100)
    outputs = tf.layers.dense(outputs, hidden_size)
    print("outputs 2 is : ", outputs)

    outputs = tf.layers.dense(outputs, 64)
    
    logits = tf.layers.dense(outputs, 2)
    print("logits is :", logits)

    onehot_labels = tf.one_hot(indices=label_ids, depth=2)

    total_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    print("total loss : ", total_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)

        train_op = optimizer.minimize(loss = total_loss, global_step=tf.train.get_global_step())

        output_spec = tf.estimator.EstimatorSpec(
                          mode=mode,
                          loss=total_loss,
                          train_op=train_op)

        return output_spec

    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(logits, axis = -1, output_type = tf.int32)
        accuracy = tf.metrics.accuracy(
            labels = label_ids,
            predictions = predictions,
        )

        #loss = tf.metrics.mean(values = per_example_loss)

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

def parse_example(serialized_example):
    name_to_features = {
        "ids1" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "ids2" : tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "labels" : tf.FixedLenFeature([], tf.int64),
    }

    example = tf.parse_single_example(serialized_example, name_to_features);

    for name in list(name_to_features.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    print("examples : ", example)
    return example


def test_data():
    d = tf.data.TFRecordDataset("output_dir/train.TFRecord.bak")
    dataset = d.map(parse_example)
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

    estimator = tf.estimator.Estimator(model_fn = simnet_model_fn, 
                                       config = run_config,
                                       params = {
                                           'batch_size' : FLAGS.train_batch_size,
                                           'num_epochs' : FLAGS.num_epochs,
                                           'hidden_units' : FLAGS.hidden_units,
                                           'learning_rate' : 0.001,
                                           'forget_bias' : 1.0,
                                       })
    print(estimator)

    if FLAGS.do_train:
        #获取训练数据
        train_examples = data_processor.get_train_examples(FLAGS.data_dir)
        train_file = os.path.join(FLAGS.output_dir, "train.TFRecord.bak")
        file_based_convert_examples_to_features(train_examples, FLAGS.max_seq_length, train_file)

        train_input_fn = file_based_input_fn_builder(
            input_file = train_file,
            seq_length = FLAGS.max_seq_length, 
            is_training = True,
            drop_remainder = True)

        print("train_input_fn:", train_input_fn)

        estimator.train(train_input_fn, max_steps = 100000)

    if FLAGS.do_eval:
        dev_examples = data_processor.get_dev_examples(FLAGS.data_dir)
        dev_file = os.path.join(FLAGS.output_dir, "eval.TFRecord.bak")
        file_based_convert_examples_to_features(dev_examples, FLAGS.max_seq_length, dev_file)

        eval_input_fn = file_based_input_fn_builder(
            input_file = dev_file,
            seq_length = FLAGS.max_seq_length, 
            is_training = True,
            drop_remainder = True)

        result = estimator.evaluate(eval_input_fn, steps = 100)



if __name__ == "__main__":
    tf.app.run()
    #test_data()

