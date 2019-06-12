# -*- coding:utf-8 -*-

import sys
import os

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

from input import InputData

# init
batch_size = 128

def source_to_seq(source_line, source_letter_to_int):
    """
    data separated by blank space 
    """
    max_seq_num = 20
    line_sp = source_line.split(' ')

    res_list = []
    for i in range(min(len(line_sp), max_seq_num)):
        res_list.append(source_letter_to_int.get(line_sp[i], source_letter_to_int['<UNK>']))

    if len(line_sp) < max_seq_num:
        res_list.append(source_letter_to_int['<PAD>'])
        #print('pad int is : ', source_letter_to_int['<PAD>'])

    return res_list

def read_ori_data(path):
    ori_list = []
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip()
            line_sp = line.split('\t')
            if len(line_sp) != 2:
                continue
            ori_list.append(line_sp[0])
    return ori_list

def load_dict(path):
    id2term = dict()
    term2id = dict()
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip()
            line_sp = line.split('\t')

            if len(line_sp) != 2:
                continue

            id = int(line_sp[0])
            term = line_sp[1]
            id2term[id] = term
            term2id[term] = id
    return id2term, term2id

# input data
#input_data = InputData('./input_data.train')
#src_list, to_list = input_data.get_input_data()

source_int_to_letter, source_letter_to_int = load_dict('src_dict')
target_int_to_letter, target_letter_to_int = load_dict('des_dict')

# predict demo
ori_list = read_ori_data('input_data.test')
#ori_list = read_ori_data('input_data.t1')
input_ids = []
for elem in ori_list:
    input_ids.append(source_to_seq(elem, source_letter_to_int))
#print(input_ids)

#input_words = 'adafad'
#input_ids = source_to_seq(input_words, source_letter_to_int)

checkpoint = "./model/epoch_6/trained_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph = loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    for i in range(len(input_ids)):
        answer_logits = sess.run(logits, {input_data: [input_ids[i]] * batch_size, 
                                      target_sequence_length: [len(input_ids[i]) + 10] * batch_size, 
                                      source_sequence_length: [len(input_ids[i]) - 1] * batch_size})[0] 

        pad = source_letter_to_int["<PAD>"] 
        print('original input is :', ori_list[i])
        print('source is ----->')
        print('original input ids:', input_ids[i])
        print('original text is : ', ''.join([source_int_to_letter[x] for x in input_ids[i]]))
        print('target is --->')
        print('target ids is : ', [x for x in answer_logits if x != pad])
        print('target text is : ', ''.join([target_int_to_letter[x] if x != 0 else ""  for x in answer_logits if x != pad]))

        print('==================>')


