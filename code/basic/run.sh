#!/bin/bash
source ~/.bashrc

#注意，等号之间不能有空格，shell语法

CUDA_VISIBLE_DEVICES=3 python_chg BERT_NER.py --do_train=true \
                                  --data_dir=./NERdata \
                                  --task_name="ner" \
                                  --bert_config_file="./chinese_L-12_H-768_A-12/bert_config.json" \
                                  --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
                                  --output_dir="./export" \
                                  --vocab_file="./chinese_L-12_H-768_A-12/vocab.txt"
