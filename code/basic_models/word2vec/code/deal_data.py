# -*- coding:utf-8 -*-

import sys
import logging
import numpy as np
#np.set_printoptions(threshold=np.inf)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InputData(object):
    def __init__(self, data_path, min_cnt, win_size, neg_num):
        """
        data_path : source data path
        min_cnt : min term feq
        """
        self.data_path = data_path
        self.min_cnt = min_cnt
        self.win_size = win_size
        self.neg_num = neg_num

        # calc term frequency
        self.term_freq = dict()

        self.sample_table = []

        # dict map
        self.term2id = dict()
        self.id2term = dict()

        self.all_lines = []

        #all pair data
        self.all_pair_data = []

        self.calc_term_freq()
        self.build_neg_sample_table()

    def calc_term_freq(self):
        """
        calc term frequency
        """
        term_freq = dict()
        with open(self.data_path) as f:
            for line in f.readlines():
                line = line.rstrip()
                line_sp = line.split(' ')
                self.all_lines.append(line_sp)
                for term in line_sp:
                    term_freq[term] = term_freq[term] + 1 if term in term_freq else 1

        #delete
        idx = 0
        for term, freq in term_freq.items():
            if freq < self.min_cnt:
                continue
            self.term2id[term] = idx
            self.id2term[idx] = term
            self.term_freq[term] = freq
            idx += 1

        #dump id
        f = open('id2term', 'w')
        for id, term in self.id2term.items():
            line = '%s\t%s\n' %(id, term)
            f.write(line)
        f.close()

    def calc_all_pair_data(self):
        """
        calc all pair u, v
        """
        for line in self.all_lines:
            line_pair = []
            term_len = len(line)
            for i, u in enumerate(line):
                if u not in self.term2id:
                    continue

                start = max(0, i - self.win_size)
                end = min(term_len, i + self.win_size + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    v = line[j]
                    if v not in self.term2id:
                        continue
                    one_list = []
                    one_list.append(self.term2id[u])
                    one_list.append(self.term2id[v])
                    neg_list = np.random.choice(self.sample_table, self.neg_num)
                    one_list.extend(neg_list)
                    line_pair.append(one_list)
                    #line_pair.append((self.term2id[u], self.term2id[v], neg_list))
            self.all_pair_data.append(line_pair)

        #dump data
        for line in self.all_pair_data:
            for one_pair in line:
                print ' '.join(str(x) for x in one_pair)


    def build_neg_sample_table(self):
        sampel_table_size = 1e8
        pow_frequency = np.array(list(self.term_freq.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sampel_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = np.array(self.sample_table)


if __name__ == '__main__':
    input_data = InputData('../data/zhihu.txt', 2, 3, 5)
    logging.info("start to run!")
    input_data.calc_all_pair_data()


