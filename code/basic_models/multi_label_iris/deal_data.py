# -*- coding:utf-8 -*-

import sys
import random

data_dict = []

type2id = dict()

num = 1

with open('iris.data') as f:
    for line in f.readlines():
        line = line.rstrip()
        line_sp = line.split(',')
        type = line_sp[-1]
        if type not in type2id:
            type2id[type] = num
            num += 1
        data_dict.append(line)

random.shuffle(data_dict)

rand_list = [ random.randint(0, 150) for i in range(30) ]

train_f = open('train.tsf', 'w')
test_f = open('test.tsf', 'w')

for i in range(150):
    dt_split = data_dict[i].split(',')
    new_line = '\t'.join(dt_split[0:4]) + '\t' + str(type2id[dt_split[-1]])

    if i in rand_list:
        test_f.write(new_line + '\n')
    else:
        train_f.write(new_line + '\n')

train_f.close()
test_f.close()


