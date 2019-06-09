# -*- coding:utf-8 -*-

import os
import tensorflow as tf

class InputData(object):
  def __init__(self, path):
    self.path = path

  def get_input_data(self):
    source_list = []
    to_list = []
    with open(self.path) as f:
      for line in f.readlines():
        line = line.rstrip()
        line_sp = line.split('\t')

        if len(line_sp) != 2:
            continue

        source_list.append(line_sp[0])
        to_list.append(line_sp[1])

    return source_list, to_list

  def extract_character_vocab(self, data_list):
    """
    """
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    word_set = set()

    for elem in special_words:
      word_set.add(elem)

    for line in data_list:
      line_sp = line.split(' ')
      for elem in line_sp:
        word_set.add(elem)
    
    word_list = list(word_set)

    id_to_vocab = { idx : word for idx, word in enumerate(word_list)}
    vocab_to_id = { word : idx for idx, word in enumerate(word_list)}
    print(id_to_vocab)
    print(vocab_to_id)

    return id_to_vocab, vocab_to_id

  def trans_to_id(self, source_list, vocab_to_id, is_target = False):
    #print(source_list)
    src_id_list = []
    for line in source_list:
      one_id_list = []
      line = line.rstrip()
      line_sp = line.split(' ')
      for word in line_sp:
        one_id_list.append(vocab_to_id.get(word, vocab_to_id['<UNK>']))

      if is_target:
        one_id_list.append(vocab_to_id['<EOS>'])

      src_id_list.append(one_id_list)
    #print(src_id_list)
    return src_id_list
    
  #def test_input_data(path):
  #  source_list, to_list = get_input_data(path)
  #  src_id2vocab, src_vocab2id = extract_character_vocab(source_list)
  #  #to_id2vocab, to_vocab2id = extract_character_vocab(to_list)
  #  trans_to_id(source_list, src_vocab2id)

if __name__ == '__main__':
  #test_input_data('./input_data.head')
  print


