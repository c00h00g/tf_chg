# -*- coding:utf-8 -*-

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

from input import InputData


epochs = 60
batch_size = 128
rnn_size = 50
num_layers = 2
encoding_embedding_size = 15
decoding_embedding_size = 15
learning_rate = 0.001

def get_encoder_layer(input_data,
                      rnn_size,
                      num_layers,
                      source_sequence_length,
                      source_vocab_size,
                      encoding_embedding_size):

  encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

  # RNN cell
  def get_lstm_cell(rnn_size):
    lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    return lstm_cell

  cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
  encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, 
                                                    sequence_length = source_sequence_length, dtype=tf.float32)
  return encoder_output, encoder_state

def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')

    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')

    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length

def process_decoder_input(data, vocab_to_int, batch_size):
  ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
  decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
  return decoder_input

def decoding_layer(target_letter_to_int, 
                   decoding_embedding_size, 
                   num_layers, 
                   rnn_size,
                   target_sequence_length, 
                   max_target_sequence_length, 
                   encoder_state, 
                   decoder_input):

    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))

    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    output_layer = Dense(target_vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer) 

        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished = True,
                                                                       maximum_iterations = max_target_sequence_length)
    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                start_tokens,
                target_letter_to_int['<EOS>'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                predicting_helper,
                encoder_state,
                output_layer)

        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                impute_finished=True,
                maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output

def seq2seq_model(input_data, targets, lr, target_sequence_length, 
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size, 
                  rnn_size, num_layers):

    _, encoder_state = get_encoder_layer(input_data, 
           rnn_size, 
           num_layers, 
           source_sequence_length,
           source_vocab_size, 
           encoding_embedding_size)

    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int, 
                                                                        decoding_embedding_size, 
                                                                        num_layers, 
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state, 
                                                                        decoder_input) 
    return training_decoder_output, predicting_decoder_output

def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i : start_i + batch_size]
        targets_batch = targets[start_i : start_i + batch_size]

        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

if __name__ == '__main__':
    #calc source_int
    input_data = InputData('./input_data.train')
    src_list, to_list = input_data.get_input_data()

    source_int_to_letter, source_letter_to_int = input_data.extract_character_vocab(src_list, 'src_dict')
    target_int_to_letter, target_letter_to_int = input_data.extract_character_vocab(to_list, 'des_dict')

    source_int = input_data.trans_to_id(src_list, source_letter_to_int)
    target_int = input_data.trans_to_id(to_list, target_letter_to_int, is_target = True)

    print('begin --------->')
    print(source_int)
    print(target_int)
    print('end--------->')

    train_graph = tf.Graph()
    with train_graph.as_default():
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
        training_decoder_output, predicting_decoder_output = seq2seq_model(input_data, 
                targets, 
                lr, 
                target_sequence_length, 
                max_target_sequence_length, 
                source_sequence_length,
                len(source_letter_to_int),
                len(target_letter_to_int),
                encoding_embedding_size, 
                decoding_embedding_size, 
                rnn_size, 
                num_layers)    

        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    targets,
                    masks)

            optimizer = tf.train.AdamOptimizer(lr)
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

    # start to train
    train_source = source_int[batch_size:]
    train_target = target_int[batch_size:]

    valid_source = source_int[:batch_size]
    valid_target = target_int[:batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                                                                                                                source_letter_to_int['<PAD>'],
                                                                                                                target_letter_to_int['<PAD>']))
    display_step = 50 
    checkpoint = "./model/trained_model.ckpt" 
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                                source_letter_to_int['<PAD>'],
                                target_letter_to_int['<PAD>'])):

                _, loss = sess.run(
                        [train_op, cost],
                        { input_data: sources_batch,
                          targets: targets_batch,
                          lr: learning_rate,
                          target_sequence_length: targets_lengths,
                          source_sequence_length: sources_lengths })

                if batch_i % display_step == 0:
                    validation_loss = sess.run(
                        [cost],
                        { input_data: valid_sources_batch,
                          targets: valid_targets_batch,
                          lr: learning_rate,
                          target_sequence_length: valid_targets_lengths,
                          source_sequence_length: valid_sources_lengths})

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                            .format(epoch_i,
                                epochs, 
                                batch_i, 
                                len(train_source) // batch_size, 
                                loss, 
                                validation_loss[0]))
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')



